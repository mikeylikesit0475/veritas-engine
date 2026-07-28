"""Tests for ``fast_check`` and the Sprint 0.5 graph.

ARCH §9: "Graph | Stub model → assert route taken and node visit order | yes".
These tests use the :class:`StubLlm` (no live calls, CLAUDE.md #6) and fake search
backends. The stub here is extended to return a canned *verdict JSON* so the schema
validation path is exercised.
"""

from __future__ import annotations

from typing import Any

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from tests.fakes import FakeToolContext
from veritas.backends_registry import clear_backends, register_backends
from veritas.graph.claim_state import (
    CLAIM_STATE_KEY,
    ClaimState,
    ClaimStatus,
    VerdictLabel,
)
from veritas.graph.fast_check import (
    FAST_CHECK_MODEL_STATE_KEY,
    FastCheckResult,
    fast_check,
)
from veritas.tools.backends import FakeSearchBackend, SearchHit
from veritas.tools.grounded_search import SEARCH_BACKEND_STATE_KEY

pytestmark = pytest.mark.asyncio


class _VerdictStubLlm(BaseLlm):
    """A stub that returns a fixed JSON verdict. For fast_check unit tests only.

    The real ``StubLlm`` in ``veritas.models.stub`` returns a plain string — fine
    for graph traversal, but ``fast_check`` parses JSON, so we need a stub that
    returns valid verdict JSON here. This keeps the unit test honest about the
    schema-validation path without a live model.
    """

    model: str = "veritas-verdict-stub"

    def __init__(self, json_response: str, **kwargs: Any) -> None:
        super().__init__(model="veritas-verdict-stub", **kwargs)
        self._json = json_response

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"veritas-verdict-stub.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> Any:  # AsyncGenerator, but typed loose to satisfy the base
        content = genai_types.Content(
            role="model",
            parts=[genai_types.Part.from_text(text=self._json)],
        )
        yield LlmResponse(content=content, turn_complete=True)


async def test_fast_check_writes_verdict_into_state() -> None:
    # The core contract: fast_check runs a search, asks the model, and writes a
    # validated Verdict into ctx.state[CLAIM_STATE_KEY].
    hits = [
        SearchHit(
            url="https://britannica.com/eiffel-tower",
            title="Eiffel Tower | Location & Facts",
            snippet="The Eiffel Tower is in Paris, France.",
        )
    ]
    model = _VerdictStubLlm(
        '{"label":"false","confidence":0.95,"rationale":"Eiffel Tower is in Paris."}'
    )
    register_backends(
        "veritas",
        **{SEARCH_BACKEND_STATE_KEY: FakeSearchBackend(hits),
           FAST_CHECK_MODEL_STATE_KEY: model},
    )
    ctx = FakeToolContext(
        {
            CLAIM_STATE_KEY: ClaimState(
                normalised_text="The Eiffel Tower is in Berlin."
            ).model_dump(mode="json"),
        }
    )
    try:
        result = await fast_check(ctx=ctx)
    finally:
        clear_backends("veritas")

    assert isinstance(result, FastCheckResult)
    assert result.verdict.label is VerdictLabel.FALSE
    assert result.verdict.confidence == 0.95
    assert "Paris" in result.verdict.rationale

    # State was persisted (ARCH §7).
    claim_state = ClaimState.from_state(ctx.state)
    assert claim_state.status is ClaimStatus.FAST_CHECKED
    assert claim_state.verdict.label is VerdictLabel.FALSE


async def test_fast_check_logs_query_verbatim() -> None:
    # ARCH §5: query logged verbatim. fast_check builds the query deterministically
    # from the claim; that exact string must be in the log.
    register_backends(
        "veritas",
        **{SEARCH_BACKEND_STATE_KEY: FakeSearchBackend([]),
           FAST_CHECK_MODEL_STATE_KEY: _VerdictStubLlm(
               '{"label":"mixed","confidence":0.3,"rationale":"no results"}')},
    )
    ctx = FakeToolContext(
        {CLAIM_STATE_KEY: ClaimState(
            normalised_text="The Eiffel Tower is in Berlin."
        ).model_dump(mode="json")}
    )
    try:
        await fast_check(ctx=ctx)
    finally:
        clear_backends("veritas")

    from veritas.tools.grounded_search import SEARCH_LOG_STATE_KEY

    assert ctx.state[SEARCH_LOG_STATE_KEY] == ["The Eiffel Tower is in Berlin."]


async def test_fast_check_raises_on_empty_model_output() -> None:
    # ADR-004: schema failures fail loudly. An empty model response is a
    # programmer/model error, not a silent "unknown" verdict.
    register_backends(
        "veritas",
        **{SEARCH_BACKEND_STATE_KEY: FakeSearchBackend([]),
           FAST_CHECK_MODEL_STATE_KEY: _VerdictStubLlm("")},
    )
    ctx = FakeToolContext(
        {CLAIM_STATE_KEY: ClaimState(normalised_text="some claim").model_dump(mode="json")}
    )
    try:
        with pytest.raises(ValueError, match="empty content"):
            await fast_check(ctx=ctx)
    finally:
        clear_backends("veritas")


async def test_fast_check_raises_on_bad_json() -> None:
    # Non-JSON model output must raise (loudly), not be coerced to a default.
    register_backends(
        "veritas",
        **{SEARCH_BACKEND_STATE_KEY: FakeSearchBackend([]),
           FAST_CHECK_MODEL_STATE_KEY: _VerdictStubLlm("I think the claim is false.")},
    )
    ctx = FakeToolContext(
        {CLAIM_STATE_KEY: ClaimState(normalised_text="some claim").model_dump(mode="json")}
    )
    try:
        with pytest.raises(ValueError, match="schema-valid JSON verdict"):
            await fast_check(ctx=ctx)
    finally:
        clear_backends("veritas")


async def test_fast_check_raises_on_bad_label() -> None:
    # JSON with an out-of-enum label is a schema failure too.
    register_backends(
        "veritas",
        **{SEARCH_BACKEND_STATE_KEY: FakeSearchBackend([]),
           FAST_CHECK_MODEL_STATE_KEY: _VerdictStubLlm(
               '{"label":"definitely","confidence":0.9,"rationale":"x"}')},
    )
    ctx = FakeToolContext(
        {CLAIM_STATE_KEY: ClaimState(normalised_text="some claim").model_dump(mode="json")}
    )
    try:
        with pytest.raises(ValueError, match="schema-valid JSON verdict"):
            await fast_check(ctx=ctx)
    finally:
        clear_backends("veritas")


async def test_fast_check_raises_without_search_backend() -> None:
    # Forgetting to register a backend is a programmer error, not silent degradation.
    ctx = FakeToolContext(
        {CLAIM_STATE_KEY: ClaimState(normalised_text="x").model_dump(mode="json")}
    )
    clear_backends("veritas")  # ensure nothing registered
    with pytest.raises(RuntimeError, match="No backends registered"):
        await fast_check(ctx=ctx)


async def test_fast_check_raises_without_claim_in_state() -> None:
    # If intake didn't run, there's no claim — fast_check should fail loudly.
    register_backends(
        "veritas",
        **{SEARCH_BACKEND_STATE_KEY: FakeSearchBackend([]),
           FAST_CHECK_MODEL_STATE_KEY: _VerdictStubLlm("{}")},
    )
    ctx = FakeToolContext({})
    try:
        with pytest.raises(RuntimeError, match="no claim text in state"):
            await fast_check(ctx=ctx)
    finally:
        clear_backends("veritas")


# --- Graph traversal test (ARCH §9: node visit order with stub model) ---


async def test_graph_runs_intake_then_fast_check_in_order() -> None:
    # The full graph: START -> intake -> triage -><route> -> fast_check -> report.
    # We drive it with the runner and assert the events we see came from the
    # expected nodes in order. This is the ARCH §9 "Graph" test category.
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    from veritas.graph.triage import TRIAGE_MODEL_STATE_KEY
    from veritas.graph.wiring import build_graph

    register_backends(
        "veritas",
        **{
            SEARCH_BACKEND_STATE_KEY: FakeSearchBackend(
                [
                    SearchHit(
                        url="https://britannica.com/eiffel-tower",
                        title="Eiffel Tower",
                        snippet="The Eiffel Tower is in Paris, France.",
                    )
                ]
            ),
            FAST_CHECK_MODEL_STATE_KEY: _VerdictStubLlm(
                '{"label":"false","confidence":0.95,"rationale":"Eiffel Tower is in Paris."}'
            ),
            TRIAGE_MODEL_STATE_KEY: _VerdictStubLlm(
                '{"path":"fast","reason":"simple claim"}'
            ),
        },
    )

    graph = build_graph()
    session_service = InMemorySessionService()  # type: ignore[no-untyped-call]
    session = await session_service.create_session(  # type: ignore[no-any-return]
        app_name="veritas", user_id="test"
    )
    runner = Runner(
        app_name="veritas",
        agent=graph,
        session_service=session_service,
    )

    message = genai_types.Content(
        role="user",
        parts=[genai_types.Part.from_text(text="The Eiffel Tower is in Berlin.")],
    )

    node_names_seen: list[str] = []
    try:
        async for event in runner.run_async(
            user_id="test", session_id=session.id, new_message=message
        ):
            name = event.node_name or event.author
            if name:
                node_names_seen.append(name)
    finally:
        clear_backends("veritas")

    # The fast path visits intake, triage, fast_check, report in order.
    assert "intake" in node_names_seen
    assert "fast_check" in node_names_seen
    assert node_names_seen.index("intake") < node_names_seen.index("fast_check")

    # The graph runs to report, so the final status is REPORTED (not FAST_CHECKED).
    from veritas.graph.claim_state import ClaimState as CS
    from veritas.graph.claim_state import ClaimStatus as CS2

    final_state = await session_service.get_session(
        app_name="veritas", user_id="test", session_id=session.id
    )
    raw = final_state.state if final_state else {}
    state_dict = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
    claim = CS.from_state(state_dict)
    assert claim.status is CS2.REPORTED
    assert claim.verdict.label is VerdictLabel.FALSE
