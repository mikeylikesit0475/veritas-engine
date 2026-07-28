"""Tests for the model factory (ADR-004, ADR-007, ADR-009).

CI-safe: exercises the stub and selection logic only. Never makes a live call — the
CI workflow asserts no model key is set, and these tests rely on that invariant.
"""

from __future__ import annotations

import pytest

from veritas.models import Backend, Tier


def test_backends_are_string_enum() -> None:
    # ADR-009: backends serialize as their literal string value.
    assert Backend.OLLAMA.value == "ollama"
    assert Backend.GEMINI.value == "gemini"
    assert Backend.STUB.value == "stub"


def test_tiers_cover_judgement_and_fanout() -> None:
    # ADR-004: two tiers. Judgement emits structured output; fan-out does not.
    assert {Tier.JUDGEMENT, Tier.FANOUT} == set(Tier)


def test_gemini_without_key_raises_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    # ADR-004 invariant: schema failures (and missing keys) surface loudly, never
    # silently. A gemini backend with no key must raise, not fall back.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("VERITAS_JUDGEMENT", "gemini")
    with pytest.raises(RuntimeError, match="no GEMINI_API_KEY"):
        from veritas.models import model_for

        model_for(Tier.JUDGEMENT)


def test_unknown_backend_raises_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    # ADR-004: bad config fails loudly, never silently degrades.
    monkeypatch.setenv("VERITAS_JUDGEMENT", "voodoo")
    with pytest.raises(ValueError, match="not a valid backend"):
        from veritas.models import model_for

        model_for(Tier.JUDGEMENT)
