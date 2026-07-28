"""Recording and replay backends for the eval harness.

**Record mode** (``VERITAS_RECORD=1``, local only): wrapping backends delegate to
real backends and append every call to an in-memory fixture, which is saved on
completion. Run via ``make record``.

**Replay mode** (default, CI): replay backends serve recorded responses, keyed by
the exact query/prompt. If a call is not in the fixture, replay raises — this is
the guarantee that CI is deterministic and offline. A missing fixture entry means
the fixture is stale, not that we silently call a live backend.

Both search and model backends follow the same Protocol as the real ones
(:class:`SearchBackend`, :class:`BaseLlm`), so the graph and server don't change.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from veritas.eval_harness.fixtures import Fixture, ModelCall, SearchCall, hash_prompt
from veritas.tools.backends import SearchBackend, SearchHit

__all__ = [
    "RecordingModel",
    "RecordingSearchBackend",
    "ReplayModel",
    "ReplaySearchBackend",
]


# --- Search: record ---


class RecordingSearchBackend:
    """Wraps a real :class:`SearchBackend`, recording every query + its hits."""

    def __init__(self, real: SearchBackend, fixture: Fixture) -> None:
        self._real = real
        self._fixture = fixture

    async def search(self, query: str) -> Sequence[SearchHit]:
        hits = list(await self._real.search(query))
        # Deduplicate by query — if the same query is asked twice, the second call
        # overwrites the first recording. That's the correct semantic: replay
        # returns the same hits regardless of call order.
        self._fixture.search_calls = [
            c for c in self._fixture.search_calls if c.query != query
        ]
        self._fixture.search_calls.append(SearchCall(query=query, hits=hits))
        return hits


# --- Search: replay ---


class ReplaySearchBackend:
    """Serves recorded search hits. Raises if a query wasn't recorded."""

    def __init__(self, fixture: Fixture) -> None:
        self._fixture = fixture

    async def search(self, query: str) -> Sequence[SearchHit]:
        hits = self._fixture.find_search(query)
        if hits is None:
            raise KeyError(
                f"ReplaySearchBackend: query not in fixture {self._fixture.name!r}: "
                f"{query!r}. Re-record with VERITAS_RECORD=1."
            )
        return list(hits)


# --- Model: record ---


class RecordingModel(BaseLlm):
    """Wraps a real :class:`BaseLlm`, recording every prompt + response."""

    model: str = "veritas-recording"

    def __init__(self, real: BaseLlm, fixture: Fixture, **kwargs: Any) -> None:
        super().__init__(model=real.model, **kwargs)
        self._real = real
        self._fixture = fixture

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r".*"]  # wraps any model

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        # Reconstruct the prompt text for hashing/recording. We join all non-thought
        # text parts from the request contents — this is what the model actually sees.
        prompt = _extract_prompt(llm_request)

        # Delegate to the real model, collecting the response text.
        response_text = ""
        async for response in self._real.generate_content_async(llm_request, stream):
            if response.content and response.content.parts:
                for part in response.content.parts:
                    if part.text and not part.thought:
                        response_text += part.text
            yield response

        # Record after yielding (we have the full response now).
        if prompt:
            self._fixture.model_calls = [
                c for c in self._fixture.model_calls if c.prompt_hash != hash_prompt(prompt)
            ]
            self._fixture.model_calls.append(
                ModelCall(
                    prompt_hash=hash_prompt(prompt),
                    prompt=prompt,
                    response=response_text,
                )
            )


# --- Model: replay ---


class ReplayModel(BaseLlm):
    """Serves recorded model responses. Raises if a prompt wasn't recorded."""

    model: str = "veritas-replay"

    def __init__(self, fixture: Fixture, **kwargs: Any) -> None:
        super().__init__(model="veritas-replay", **kwargs)
        self._fixture = fixture

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"veritas-replay.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        prompt = _extract_prompt(llm_request)
        response_text = self._fixture.find_model(prompt)
        if response_text is None:
            raise KeyError(
                f"ReplayModel: prompt not in fixture {self._fixture.name!r} "
                f"(hash={hash_prompt(prompt)[:12]}). Re-record with VERITAS_RECORD=1."
            )
        content = genai_types.Content(
            role="model",
            parts=[genai_types.Part.from_text(text=response_text)],
        )
        yield LlmResponse(content=content, turn_complete=True)


def _extract_prompt(llm_request: LlmRequest) -> str:
    """Join all non-thought text parts from the request into a single string."""
    parts: list[str] = []
    for content in llm_request.contents:
        if content and content.parts:
            for part in content.parts:
                if part.text and not part.thought:
                    parts.append(part.text)
    return "\n".join(parts)
