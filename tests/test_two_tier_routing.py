"""Tests for two-tier model routing — ADR-004, ADR-009.

Verifies that the factory produces the right backend for each tier, that env
overrides work, and that the fan-out tier can be swapped independently of the
judgement tier. CI-safe: uses the stub where a live backend would be needed.
"""

from __future__ import annotations

import pytest

from veritas.models import Backend, Tier, model_for


def test_judgement_and_fanout_are_distinct_tiers() -> None:
    # ADR-004: two tiers. Judgement emits structured output; fan-out does not.
    assert Tier.JUDGEMENT is not Tier.FANOUT


def test_default_backend_is_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    # ADR-009: Ollama is the default on this machine.
    monkeypatch.delenv("VERITAS_JUDGEMENT", raising=False)
    monkeypatch.delenv("VERITAS_FANOUT", raising=False)
    # We can't call model_for() without hitting a live backend, but we can
    # verify the env-reading logic returns the right Backend enum.
    from veritas.models import _env_backend

    assert _env_backend("VERITAS_JUDGEMENT") is Backend.OLLAMA
    assert _env_backend("VERITAS_FANOUT") is Backend.OLLAMA


def test_fanout_tier_can_be_swapped_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    # ADR-009: the two tiers are independently configurable. Setting FANOUT=gemini
    # should not change JUDGEMENT.
    monkeypatch.setenv("VERITAS_FANOUT", "gemini")
    monkeypatch.delenv("VERITAS_JUDGEMENT", raising=False)
    from veritas.models import _env_backend

    assert _env_backend("VERITAS_FANOUT") is Backend.GEMINI
    assert _env_backend("VERITAS_JUDGEMENT") is Backend.OLLAMA


def test_judgement_tier_can_be_swapped_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERITAS_JUDGEMENT", "gemini")
    monkeypatch.delenv("VERITAS_FANOUT", raising=False)
    from veritas.models import _env_backend

    assert _env_backend("VERITAS_JUDGEMENT") is Backend.GEMINI
    assert _env_backend("VERITAS_FANOUT") is Backend.OLLAMA


def test_custom_model_names_are_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    # VERITAS_FANOUT_MODEL overrides the default model name.
    monkeypatch.setenv("VERITAS_FANOUT_MODEL", "qwen3-coder:latest")
    from veritas.models import _env_backend, _env_model

    backend = _env_backend("VERITAS_FANOUT")
    model = _env_model("VERITAS_FANOUT_MODEL", Tier.FANOUT, backend)
    # Ollama models get the ollama_chat/ prefix automatically.
    assert model == "ollama_chat/qwen3-coder:latest"


def test_gemini_fanout_without_key_raises_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    # ADR-004: missing key fails loudly, never silently degrades.
    monkeypatch.setenv("VERITAS_FANOUT", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="no GEMINI_API_KEY"):
        model_for(Tier.FANOUT)
