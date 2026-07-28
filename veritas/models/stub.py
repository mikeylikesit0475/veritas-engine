"""Stub LLM for CI — never a live model call (CLAUDE.md non-negotiable #6).

Returns a canned response so graph traversal tests can assert node visit order without
any network or model dependency. The stub is intentionally minimal: it does not honour
schemas, because CI never exercises schema reliability — that is a live-eval concern
(Sprint 3) and recording it in CI would be a category error.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

__all__ = ["STUB_RESPONSE", "StubLlm"]

STUB_RESPONSE = "[stub] veritas has nothing to say in CI — no live model calls."


class StubLlm(BaseLlm):
    """A deterministic, offline LLM for CI. Yields one canned ``LlmResponse``."""

    model: str = "veritas-stub"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model="veritas-stub", **kwargs)

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"veritas-stub.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        content = genai_types.Content(
            role="model",
            parts=[genai_types.Part.from_text(text=STUB_RESPONSE)],
        )
        yield LlmResponse(content=content, turn_complete=True)
