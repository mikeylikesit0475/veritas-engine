"""Backend protocols for the tools.

Tools depend on *protocols*, not concrete clients. This is the seam that lets tests
inject fakes (ARCH §9: "Unit | Tools against fake search/fetch | yes") and lets the
server inject real HTTP backends at runtime. CI never sees a real backend
(CLAUDE.md non-negotiable #6).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

__all__ = [
    "FakeFetchBackend",
    "FakeSearchBackend",
    "FetchBackend",
    "FetchedPage",
    "SearchBackend",
    "SearchHit",
]


class SearchHit(BaseModel):
    """One search result. ``snippet`` is the unmodified search-engine excerpt."""

    url: str
    title: str
    snippet: str


@runtime_checkable
class SearchBackend(Protocol):
    """Read-only search backend. Implementations must be safe to call concurrently."""

    async def search(self, query: str) -> Sequence[SearchHit]:
        """Return results for ``query``. Order is backend-defined (relevance)."""
        ...


class FetchedPage(BaseModel):
    """Raw fetched page, *before* safety guards strip and cap it.

    ``content`` is whatever the server returned as text/html; the tool applies the
    ARCH §5 guards (script strip, size cap, delimited preamble) before the model sees it.
    """

    url: str
    content: str
    status: int
    fetched_at: float  # unix epoch, seconds


@runtime_checkable
class FetchBackend(Protocol):
    """Read-only HTTP fetch backend. Implementations must enforce a timeout."""

    async def fetch(self, url: str, *, timeout_s: float) -> FetchedPage:
        """Fetch ``url`` and return the raw page. Must respect ``timeout_s``."""
        ...


# --- Fakes for tests (ARCH §9: tools tested against fakes) ---


class FakeSearchBackend:
    """In-memory search backend. Yields whatever it was seeded with, in order."""

    def __init__(self, hits: Sequence[SearchHit] | None = None) -> None:
        self._hits: list[SearchHit] = list(hits or [])
        # ARCH §5: query logged verbatim to the event stream. Tests assert on this.
        self.queries: list[str] = []

    async def search(self, query: str) -> Sequence[SearchHit]:
        self.queries.append(query)
        return list(self._hits)


class FakeFetchBackend:
    """In-memory fetch backend. Yields a canned page for any URL.

    For per-URL control, pass a ``pages: dict[str, FetchedPage]``. Unknown URLs fall
    back to ``default`` (a 200 with empty body if not provided).
    """

    def __init__(
        self,
        pages: dict[str, FetchedPage] | None = None,
        default: FetchedPage | None = None,
    ) -> None:
        self._pages = dict(pages or {})
        self._default = default or FetchedPage(
            url="<unknown>", content="", status=200, fetched_at=0.0
        )
        self.fetches: list[str] = []

    async def fetch(self, url: str, *, timeout_s: float) -> FetchedPage:
        self.fetches.append(url)
        page = self._pages.get(url, self._default)
        # Return a copy with the requested url so tests can match on it.
        return page.model_copy(update={"url": url})
