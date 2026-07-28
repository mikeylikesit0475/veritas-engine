"""Unit tests for :class:`GroundedSearch` against a fake backend (ARCH §9)."""

from __future__ import annotations

import pytest

from tests.fakes import FakeToolContext
from veritas.tools.backends import FakeSearchBackend, SearchHit
from veritas.tools.grounded_search import (
    MAX_HITS,
    SEARCH_BACKEND_STATE_KEY,
    SEARCH_LOG_STATE_KEY,
    GroundedSearch,
)

pytestmark = pytest.mark.asyncio


async def test_search_logs_query_verbatim() -> None:
    # ARCH §5: "query logged verbatim to the event stream". The log must record
    # the exact string the model asked for, not a normalised version.
    hits = [SearchHit(url="https://e.com/x", title="X", snippet="snip")]
    backend = FakeSearchBackend(hits)
    ctx = FakeToolContext({SEARCH_BACKEND_STATE_KEY: backend})

    await GroundedSearch().run_async(
        args={"query": "Capital of  France?? "}, tool_context=ctx
    )

    assert ctx.state[SEARCH_LOG_STATE_KEY] == ["Capital of  France?? "]


async def test_search_returns_hits_as_model_dump() -> None:
    # The tool returns a dict (model_dump) because ADK serialises it into the
    # FunctionResponse. We assert the shape, not the ADK wrapper.
    hits = [
        SearchHit(url="https://e.com/a", title="A", snippet="sa"),
        SearchHit(url="https://e.com/b", title="B", snippet="sb"),
    ]
    ctx = FakeToolContext({SEARCH_BACKEND_STATE_KEY: FakeSearchBackend(hits)})

    out = await GroundedSearch().run_async(args={"query": "q"}, tool_context=ctx)

    assert out["query"] == "q"
    assert out["truncated"] is False
    assert len(out["hits"]) == 2
    assert out["hits"][0]["url"] == "https://e.com/a"


async def test_search_caps_results_at_max_hits() -> None:
    # ARCH §5: result cap. A backend returning >MAX_HITS must be truncated, and
    # the ``truncated`` flag set so the model knows.
    hits = [SearchHit(url=f"https://e.com/{i}", title=str(i), snippet="s") for i in range(MAX_HITS + 5)]
    ctx = FakeToolContext({SEARCH_BACKEND_STATE_KEY: FakeSearchBackend(hits)})

    out = await GroundedSearch().run_async(args={"query": "q"}, tool_context=ctx)

    assert len(out["hits"]) == MAX_HITS
    assert out["truncated"] is True


async def test_search_appends_to_existing_log() -> None:
    # The log is append-only across calls within a session.
    ctx = FakeToolContext(
        {
            SEARCH_BACKEND_STATE_KEY: FakeSearchBackend([]),
            SEARCH_LOG_STATE_KEY: ["prior query"],
        }
    )

    await GroundedSearch().run_async(args={"query": "new query"}, tool_context=ctx)

    assert ctx.state[SEARCH_LOG_STATE_KEY] == ["prior query", "new query"]


async def test_search_rejects_empty_query_with_validation_error() -> None:
    # Pydantic enforces min_length=1 — a model that sends an empty query gets a
    # ValidationError, not a silent empty search.
    from pydantic import ValidationError

    ctx = FakeToolContext({SEARCH_BACKEND_STATE_KEY: FakeSearchBackend([])})
    with pytest.raises(ValidationError):
        await GroundedSearch().run_async(args={"query": ""}, tool_context=ctx)


async def test_search_raises_when_no_backend_injected() -> None:
    # The server must inject a backend. Forgetting to is a programmer error, not
    # a silent degradation — it raises loudly (ADR-004 spirit).
    ctx = FakeToolContext({})
    with pytest.raises(RuntimeError, match="No search backend"):
        await GroundedSearch().run_async(args={"query": "q"}, tool_context=ctx)
