"""``GroundedSearch`` tool — ARCH §5.

Guards enforced here:
- query logged verbatim to the event stream (``tool_context.state`` append)
- result cap (``MAX_HITS``)

Arguments are a pydantic model; no ``dict[str, Any]`` crosses the model boundary
(CLAUDE.md conventions). The backend is injected via ``tool_context`` state so tests
can pass a fake and CI never touches a live search API.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents.context import Context as ToolContext
from google.adk.tools.base_tool import BaseTool
from pydantic import BaseModel, Field

from veritas.tools.backends import SearchBackend, SearchHit

__all__ = [
    "MAX_HITS",
    "SEARCH_BACKEND_STATE_KEY",
    "GroundedSearch",
    "GroundedSearchArgs",
    "GroundedSearchResult",
]

# ARCH §5: result cap. A search that returns 50 hits would blow the context window
# before the model ever reasons about them.
MAX_HITS = 5

# The state key under which the tool records verbatim queries. The event stream is
# the source of truth (ARCH §7: claim state is a projection of the event stream).
SEARCH_LOG_STATE_KEY = "veritas.search.queries"
# The state key under which the injected backend lives. The server sets this before
# the run; tests set it in the fake session.
SEARCH_BACKEND_STATE_KEY = "veritas.search.backend"


class GroundedSearchArgs(BaseModel):
    """Arguments for :class:`GroundedSearch`. The model fills this."""

    query: str = Field(
        ..., min_length=1, max_length=500, description="The search query, verbatim."
    )


class GroundedSearchResult(BaseModel):
    """Structured result returned to the runner (and thence to the model)."""

    query: str
    hits: list[SearchHit] = Field(default_factory=list)
    truncated: bool = False


class GroundedSearch(BaseTool):
    """Grounded web search. Non-mutating, read-only, query-logged."""

    def __init__(self) -> None:
        super().__init__(
            name="grounded_search",
            description=(
                "Search the web for evidence. Use a precise, factual query. "
                "Returns up to "
                f"{MAX_HITS} results with url, title, and a short snippet."
            ),
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        # Boundary: validate the raw dict into a pydantic model immediately. Nothing
        # past this line ever sees a dict[str, Any] (CLAUDE.md conventions).
        parsed = GroundedSearchArgs.model_validate(args)

        # ARCH §5: query logged verbatim to the event stream, before the backend
        # call so the log is present even if the call throws.
        log: list[str] = tool_context.state.get(SEARCH_LOG_STATE_KEY, [])
        log = [*log, parsed.query]
        tool_context.state[SEARCH_LOG_STATE_KEY] = log

        backend = _get_backend(tool_context)
        hits = list(await backend.search(parsed.query))

        truncated = len(hits) > MAX_HITS
        if truncated:
            hits = hits[:MAX_HITS]

        return GroundedSearchResult(
            query=parsed.query, hits=hits, truncated=truncated
        ).model_dump()


def _get_backend(tool_context: ToolContext) -> SearchBackend:
    """Pull the injected backend from state, asserting the protocol holds."""
    backend = tool_context.state.get(SEARCH_BACKEND_STATE_KEY)
    if backend is None:
        raise RuntimeError(
            f"No search backend in state[{SEARCH_BACKEND_STATE_KEY!r}]. The server "
            "must inject one before the run; tests must inject a fake."
        )
    if not isinstance(backend, SearchBackend):  # Protocol, runtime_checkable
        raise TypeError(
            f"State[{SEARCH_BACKEND_STATE_KEY!r}] is not a SearchBackend: "
            f"got {type(backend).__name__}"
        )
    return backend
