"""JSON parsing helpers for model output — ADR-004.

Local models (Ollama) often wrap JSON in markdown code fences (`````json ... `````)
even when ``response_mime_type="application/json"`` is set. ADR-004 predicted this:
"local and Ollama-cloud models have been unreliable at honouring JSON schema in
structured output." This module strips the fences before parsing so the schema
validation in each node receives clean JSON.

The stripping is deliberately simple: if the text starts with `````json`` (or
just ```````), strip the first and last fence lines. This is not a general
markdown parser — it's a pragmatic defence for the one thing Ollama does wrong.
"""

from __future__ import annotations

import re

__all__ = ["parse_json_lenient", "strip_markdown_fences"]

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from ``text`` if present."""
    match = _FENCE_RE.match(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_json_lenient(text: str) -> str:
    """Strip fences and return clean JSON text. Alias for readability."""
    return strip_markdown_fences(text)
