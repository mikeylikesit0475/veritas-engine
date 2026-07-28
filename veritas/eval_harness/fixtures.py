"""Fixture format for record/replay (ARCH §6, CLAUDE.md #6).

A fixture is a JSON file representing one claim's worth of recorded backend calls.
The structure is intentionally flat and human-readable — an interviewer should be
able to open one and see exactly what the model was asked and what it said.

Structure::

    {
      "claim": "The Eiffel Tower is in Berlin.",
      "search_calls": [
        {"query": "The Eiffel Tower is in Berlin.", "hits": [...]}
      ],
      "model_calls": [
        {"prompt_hash": "sha256:...", "prompt": "...", "response": "..."}
      ],
      "verdict": {"label": "false", "confidence": 1.0, "rationale": "..."}
    }

Replay matches on:
- search: exact query string
- model: exact prompt string (or its sha256, to handle very long prompts)

The fixture is the single source of truth for what CI replays. Recording is a
local-only operation; CI never writes fixtures.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from veritas.graph.claim_state import Verdict
from veritas.tools.backends import SearchHit

__all__ = ["FIXTURES_DIR", "Fixture", "hash_prompt", "load_fixture", "save_fixture"]

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "evals" / "fixtures"


class SearchCall(BaseModel):
    query: str
    hits: list[SearchHit] = Field(default_factory=list)


class ModelCall(BaseModel):
    prompt_hash: str
    prompt: str
    response: str


class Fixture(BaseModel):
    """One recorded claim's backend calls. The replay source of truth."""

    name: str
    claim: str
    search_calls: list[SearchCall] = Field(default_factory=list)
    model_calls: list[ModelCall] = Field(default_factory=list)
    verdict: Verdict = Field(default_factory=Verdict)  # type: ignore[arg-type]

    def find_search(self, query: str) -> list[SearchHit] | None:
        """Return recorded hits for ``query``, or None if not recorded."""
        for call in self.search_calls:
            if call.query == query:
                return call.hits
        return None

    def find_model(self, prompt: str) -> str | None:
        """Return recorded response for ``prompt``, or None if not recorded."""
        ph = hash_prompt(prompt)
        for call in self.model_calls:
            if call.prompt_hash == ph:
                return call.response
        return None


def hash_prompt(prompt: str) -> str:
    """Stable hash for matching model calls. SHA-256, prefix-stripped."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def fixture_path(name: str) -> Path:
    return FIXTURES_DIR / f"{name}.json"


def load_fixture(name: str) -> Fixture:
    """Load a fixture by name from ``evals/fixtures/<name>.json``."""
    path = fixture_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return Fixture.model_validate_json(path.read_text(encoding="utf-8"))


def save_fixture(fixture: Fixture) -> Path:
    """Save a fixture. Used by record mode only — CI never calls this."""
    path = fixture_path(fixture.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(fixture.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
