"""Tests for the report node + citation validator — ARCH §3 node 13, ARCH §5 #1.

> Test proves an unresolvable citation raises.

ARCH §5 safety invariant #1: "No agent output reaches the report without a
resolvable citation. Enforced in the report validator, not by instruction."

These tests prove the validator catches:
- A dangling [cite:N] that has no matching stored citation → raises
- A report where all citations resolve → passes
- The render produces markdown with inline citations
"""

from __future__ import annotations

import pytest

from tests.fakes import FakeToolContext
from veritas.graph.arbiter import ArbiterResult, SubClaimVerdict
from veritas.graph.claim_state import ClaimState, ClaimStatus, Verdict, VerdictLabel
from veritas.graph.fan_out import AgentFinding, FanOutResult
from veritas.graph.report import (
    CITATION_STORE_STATE_KEY,
    Citation,
    CitationError,
    CitationStore,
    ReportResult,
    build_citation_store_from_findings,
    render_report,
    report,
    validate_citations,
)


def _arbiter_result() -> ArbiterResult:
    return ArbiterResult(
        sub_claim_verdicts=[
            SubClaimVerdict(
                sub_claim_id="sc0",
                label=VerdictLabel.TRUE,
                confidence=0.9,
                rationale="Confirmed by Wikipedia [cite:0]",
            ),
            SubClaimVerdict(
                sub_claim_id="sc1",
                label=VerdictLabel.FALSE,
                confidence=0.95,
                rationale="Eiffel Tower is in Paris [cite:1]",
            ),
        ],
        overall_verdict=Verdict(
            label=VerdictLabel.FALSE,
            confidence=0.92,
            rationale="The Eiffel Tower is in Paris, not Berlin.",
        ),
    )


def _citation_store() -> CitationStore:
    return CitationStore(
        citations=[
            Citation(
                id=0,
                url="https://en.wikipedia.org/wiki/Eiffel_Tower",
                title="Eiffel Tower - Wikipedia",
                source="search",
                sub_claim_id="sc0",
                agent_name="steelman",
            ),
            Citation(
                id=1,
                url="https://britannica.com/eiffel-tower",
                title="Eiffel Tower | Britannica",
                source="search",
                sub_claim_id="sc1",
                agent_name="skeptic",
            ),
        ]
    )


def test_validator_passes_when_all_citations_resolve() -> None:
    store = _citation_store()
    markdown = "The tower exists [cite:0] and is in Paris [cite:1]."
    ids = validate_citations(markdown, store)
    assert ids == [0, 1]


def test_validator_raises_on_dangling_citation() -> None:
    # THE test: a [cite:N] that doesn't resolve to any stored citation must raise.
    store = _citation_store()  # has ids 0 and 1
    markdown = "Some claim [cite:0] and a dangling one [cite:99]."
    with pytest.raises(CitationError, match="unresolvable citation"):
        validate_citations(markdown, store)


def test_validator_raises_on_empty_store() -> None:
    store = CitationStore(citations=[])
    markdown = "Claim with citation [cite:0]."
    with pytest.raises(CitationError, match="unresolvable citation"):
        validate_citations(markdown, store)


def test_validator_handles_no_citations_in_text() -> None:
    store = _citation_store()
    markdown = "A claim with no citations."
    ids = validate_citations(markdown, store)
    assert ids == []


def test_render_report_produces_markdown_with_citations() -> None:
    claim_state = ClaimState(normalised_text="The Eiffel Tower is in Berlin.")
    arbiter = _arbiter_result()
    store = _citation_store()

    result = render_report(claim_state, arbiter, store)

    assert isinstance(result, ReportResult)
    assert "# Veritas Report" in result.markdown
    assert "false" in result.markdown.lower()
    assert "92%" in result.markdown
    assert result.citation_count == 2
    # Citations are present in the markdown.
    assert "[cite:0]" in result.markdown
    assert "[cite:1]" in result.markdown


def test_render_report_raises_on_dangling_citation_in_verdict() -> None:
    # If the arbiter's rationale cites a non-existent citation, the render must
    # fail — no agent output reaches the report without a resolvable citation.
    claim_state = ClaimState(normalised_text="Some claim.")
    arbiter = ArbiterResult(
        sub_claim_verdicts=[
            SubClaimVerdict(
                sub_claim_id="sc0",
                label=VerdictLabel.TRUE,
                confidence=0.9,
                rationale="Confirmed [cite:99]",  # doesn't exist
            ),
        ],
        overall_verdict=Verdict(
            label=VerdictLabel.TRUE, confidence=0.9, rationale="Yes [cite:99]"
        ),
    )
    store = CitationStore(citations=[])  # empty — no citation 99

    with pytest.raises(CitationError, match="unresolvable citation"):
        render_report(claim_state, arbiter, store)


def test_build_citation_store_extracts_urls_from_findings() -> None:
    findings = [
        AgentFinding(
            sub_claim_id="sc0",
            agent_name="skeptic",
            findings="Found evidence at https://example.com/page that refutes this.",
            elapsed_s=0.1,
        ),
        AgentFinding(
            sub_claim_id="sc0",
            agent_name="steelman",
            findings="No support found. See https://other.com/source",
            elapsed_s=0.1,
        ),
    ]
    store = build_citation_store_from_findings(findings)
    assert len(store.citations) == 2
    assert store.citations[0].url == "https://example.com/page"
    assert store.citations[1].url == "https://other.com/source"


def test_build_citation_store_deduplicates_by_url() -> None:
    findings = [
        AgentFinding(
            sub_claim_id="sc0", agent_name="skeptic",
            findings="See https://example.com/x", elapsed_s=0.1,
        ),
        AgentFinding(
            sub_claim_id="sc1", agent_name="steelman",
            findings="Also https://example.com/x", elapsed_s=0.1,
        ),
    ]
    store = build_citation_store_from_findings(findings)
    assert len(store.citations) == 1  # deduplicated


async def test_report_node_renders_and_updates_state() -> None:
    # End-to-end: the report() node takes arbiter + fanout state, renders, and
    # writes REPORTED status.
    ctx = FakeToolContext({
        "veritas.claim.state": ClaimState(
            normalised_text="The Eiffel Tower is in Berlin.",
            status=ClaimStatus.ARBITERED,
        ).model_dump(mode="json"),
        "veritas.arbiter.result": _arbiter_result().model_dump(mode="json"),
        "veritas.fanout.result": FanOutResult(
            findings=[
                AgentFinding(
                    sub_claim_id="sc0", agent_name="steelman",
                    findings="Wikipedia confirms: https://en.wikipedia.org/wiki/Eiffel_Tower",
                    elapsed_s=0.1,
                ),
                AgentFinding(
                    sub_claim_id="sc1", agent_name="skeptic",
                    findings="It's in Paris: https://britannica.com/eiffel-tower",
                    elapsed_s=0.1,
                ),
            ]
        ).model_dump(mode="json"),
    })

    result = await report(ctx=ctx)

    assert "# Veritas Report" in result.markdown
    assert result.citation_count >= 2

    claim_state = ClaimState.from_state(ctx.state)
    assert claim_state.status is ClaimStatus.REPORTED
    assert CITATION_STORE_STATE_KEY in ctx.state
