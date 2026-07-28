"""Veritas tools — typed pydantic args, never ``dict[str, Any]`` past the boundary.

ARCH §5 safety invariants live here. The two tools in this sprint:

- :class:`GroundedSearch` — query logged verbatim to the event stream; result cap.
- :class:`FetchPage` — domain denylist, 10s timeout, 200 KB cap, script-stripped,
  wrapped in a delimited data-not-instruction block (ADR-006).

``PythonSandbox`` and ``CitationStore`` land in later sprints.
"""

from __future__ import annotations

from veritas.tools.backends import (
    FakeFetchBackend,
    FakeSearchBackend,
    FetchBackend,
    FetchedPage,
    SearchBackend,
    SearchHit,
)
from veritas.tools.fetch_page import FetchPage, FetchPageArgs, FetchPageResult
from veritas.tools.grounded_search import (
    GroundedSearch,
    GroundedSearchArgs,
    GroundedSearchResult,
)

__all__ = [
    "FakeFetchBackend",
    "FakeSearchBackend",
    "FetchBackend",
    "FetchPage",
    "FetchPageArgs",
    "FetchPageResult",
    "FetchedPage",
    "GroundedSearch",
    "GroundedSearchArgs",
    "GroundedSearchResult",
    "SearchBackend",
    "SearchHit",
]
