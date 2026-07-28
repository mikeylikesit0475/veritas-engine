"""Two-tier model factory — ADR-004, ADR-007, ADR-009.

Backends:
    - ``ollama`` (default on this machine) — routes via ADK's ``LiteLlm`` bridge to a
      local or ``:cloud`` Ollama model. No external key required.
    - ``gemini`` — the Gemini API. Requires ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.

Selection is environment-driven so the same code runs in both environments:

    VERITAS_JUDGEMENT   ollama | gemini   (default: ollama)
    VERITAS_FANOUT      ollama | gemini   (default: ollama)

Model names are config, not code:

    VERITAS_JUDGEMENT_MODEL  e.g. glm-5.1:cloud, gemini-2.5-flash
    VERITAS_FANOUT_MODEL     e.g. laguna-s-32k:latest, gemini-2.5-flash

CI never calls a live model (CLAUDE.md non-negotiable #6). The stub backend lives in
``veritas.models.stub`` and is selected when no live backend is configured — i.e. always
in CI. A live backend selected without its key raises loudly, never silently (ADR-004).
"""

from __future__ import annotations

import os
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.adk.models.base_llm import BaseLlm

__all__ = ["Backend", "Tier", "model_for"]


class Backend(StrEnum):
    """A model backend. ``StrEnum`` so values serialize as the literal string."""

    OLLAMA = "ollama"
    GEMINI = "gemini"
    STUB = "stub"  # CI only — never a live call


class Tier(StrEnum):
    """ADR-004 tier. Judgement emits structured output; fan-out does not."""

    JUDGEMENT = "judgement"
    FANOUT = "fanout"


_DEFAULTS: dict[Tier, dict[Backend, str]] = {
    Tier.JUDGEMENT: {
        Backend.OLLAMA: "ollama_chat/glm-5.1:cloud",
        Backend.GEMINI: "gemini-2.5-flash",
    },
    Tier.FANOUT: {
        Backend.OLLAMA: "ollama_chat/laguna-s-32k:latest",
        Backend.GEMINI: "gemini-2.5-flash",
    },
}


def _env_backend(var: str) -> Backend:
    """Read a backend selector from the environment. Unknown values raise loudly."""
    raw = os.environ.get(var, "").strip().lower()
    if not raw:
        return Backend.OLLAMA  # ADR-009: Ollama is the default on this machine.
    try:
        return Backend(raw)
    except ValueError as e:
        raise ValueError(
            f"{var}={raw!r} is not a valid backend. Use one of: "
            f"{', '.join(b.value for b in Backend if b is not Backend.STUB)}."
        ) from e


def _env_model(var: str, tier: Tier, backend: Backend) -> str:
    """Read a model name, falling back to the tier/backend default."""
    raw = os.environ.get(var, "").strip()
    if raw:
        # Ollama models via LiteLLM need the ollama_chat/ prefix for chat models.
        if backend is Backend.OLLAMA and not raw.startswith(("ollama/", "ollama_chat/")):
            return f"ollama_chat/{raw}"
        return raw
    return _DEFAULTS[tier][backend]


def model_for(tier: Tier) -> BaseLlm:
    """Return the ADK ``BaseLlm`` for a tier, selected from the environment.

    The factory is the single seam between the rest of the package and ADK's model
    classes (ADR-001). An upstream API change touches this function, not the graph.
    """
    if tier is Tier.JUDGEMENT:
        backend = _env_backend("VERITAS_JUDGEMENT")
        model = _env_model("VERITAS_JUDGEMENT_MODEL", tier, backend)
    else:
        backend = _env_backend("VERITAS_FANOUT")
        model = _env_model("VERITAS_FANOUT_MODEL", tier, backend)

    if backend is Backend.OLLAMA:
        return _ollama_llm(model)
    if backend is Backend.GEMINI:
        return _gemini_llm(model)
    # pragma: no cover - stub selected only when explicitly requested in CI
    return _stub_llm()


def _ollama_llm(model: str) -> BaseLlm:
    # Local import: LiteLlm pulls in litellm, which is heavy. Keep it out of the
    # module import graph so CI/stub paths don't pay for it.
    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(model=model)


def _gemini_llm(model: str) -> BaseLlm:
    # ADR-004: the judgement tier's structured-output reliability depends on a real
    # Gemini key being present. Fail loudly if it's missing — never silently degrade.
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError(
            "VERITAS_JUDGEMENT/FANOUT=gemini but no GEMINI_API_KEY or GOOGLE_API_KEY is "
            "set. Either set a key or switch to VERITAS_*=ollama (ADR-009)."
        )
    from google.adk.models.google_llm import Gemini

    return Gemini(model=model)


def _stub_llm() -> BaseLlm:
    from veritas.models.stub import StubLlm

    return StubLlm()
