"""Real HTTP backends for the live demo.

These are *not* used in CI — CI injects fakes (CLAUDE.md #6). The server wires these
into session state at runtime so the same graph that runs against fakes in tests runs
against real backends in the demo.

Sprint 0.5 ships:

- :class:`HttpSearchBackend` — queries DuckDuckGo's HTML endpoint (no API key, no
  billing) and parses the result cards. This is intentionally simple: it is *not* a
  production search integration. It exists so the demo can run end-to-end on this
  machine with no external key (ADR-009). Sprint 1 will likely swap it for Google
  Search grounding when a Gemini key is available, behind the same
  :class:`SearchBackend` protocol.
- :class:`HttpFetchBackend` — ``httpx.AsyncClient`` with a timeout, returning a
  :class:`FetchedPage`. The :class:`FetchPage` tool applies all the ARCH §5 guards
  (denylist, size cap, script strip) on top of whatever this returns.

Honest caveat: DuckDuckGo's HTML layout can change. If it does, the live demo breaks
but CI stays green (fakes). That is the correct failure mode for a portfolio PoC.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from html import unescape

import httpx

from veritas.tools.backends import FetchedPage, SearchHit

__all__ = ["HttpFetchBackend", "HttpSearchBackend"]

# DuckDuckGo HTML endpoint. The ``kd=-1`` disables locale biasing; ``kl=us-en``
# pins region so demo results are reproducible.
_DDHTML_URL = "https://html.duckduckgo.com/html/"
_DDHTML_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 veritas-engine/0.1"
    )
}

# Result-card parser. DuckDuckGo's HTML wraps each result in
# ``<a class="result__a" href="...">Title</a>`` with a
# ``<a class="result__snippet" ...>snippet</a>``. These regexes are deliberately
# loose — see the module docstring's caveat.
_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _parse_ddg_html(html: str) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for url, title, snippet in _RESULT_RE.findall(html):
        # DuckDuckGo redirects via ``//duckduckgo.com/l/?uddg=<encoded>``. Decode it.
        m = re.search(r"uddg=([^&]+)", url)
        if m:
            from urllib.parse import unquote

            url = unquote(m.group(1))
        title = unescape(_TAG_RE.sub("", title)).strip()
        snippet = unescape(_TAG_RE.sub("", snippet)).strip()
        if url and title:
            hits.append(SearchHit(url=url, title=title, snippet=snippet))
    return hits


class HttpSearchBackend:
    """Read-only search via DuckDuckGo HTML. No API key. For the live demo only."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(10.0), headers=_DDHTML_HEADERS
        )
        self._owns_client = client is None

    async def search(self, query: str) -> Sequence[SearchHit]:
        try:
            resp = await self._client.post(
                _DDHTML_URL,
                data={"q": query, "kd": "-1", "kl": "us-en"},
            )
            resp.raise_for_status()
            return _parse_ddg_html(resp.text)
        except httpx.HTTPError:
            # A search failure should not crash the graph — return empty so the
            # model can say "no evidence found". The event stream still has the
            # verbatim query (logged before the backend call in the tool/node).
            return []

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class HttpFetchBackend:
    """Read-only HTTP fetch via httpx. Respects the timeout passed by the tool."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
            headers={"User-Agent": "veritas-engine/0.1 (portfolio PoC)"},
        )
        self._owns_client = client is None

    async def fetch(self, url: str, *, timeout_s: float) -> FetchedPage:
        # httpx's timeout is set per-request here, not just on the client, so the
        # tool's TIMEOUT_SECONDS is the one that actually binds.
        resp = await self._client.get(
            url, timeout=httpx.Timeout(timeout_s), follow_redirects=True
        )
        return FetchedPage(
            url=str(resp.url),
            content=resp.text,
            status=resp.status_code,
            fetched_at=time.time(),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
