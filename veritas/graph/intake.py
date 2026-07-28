"""``intake`` node — ARCH §3, node 1.

Pure Python. No model. Responsibilities:
- Normalise whitespace.
- Strip prompt-injection markers (the same defence as ADR-006, applied to the *claim
  itself* — a claim is user input, not trusted text).
- Reject claims > 2000 chars.
- Write a :class:`ClaimState` into ``ctx.state``.

Control flow stays in Python (ADR-005). The node is a plain async function bound into
the graph via ADK's :class:`FunctionNode`; the function signature is the contract.
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel

from veritas.graph.claim_state import (
    CLAIM_STATE_KEY,
    ClaimState,
    ClaimStatus,
)

__all__ = ["INTAKE_MAX_CHARS", "IntakeResult", "intake"]

# ARCH §3: reject >2000 chars.
INTAKE_MAX_CHARS = 2000

# Prompt-injection markers we strip from the claim text. These are the *structural*
# markers (role tags, delimiter fakes) — a plain-text "ignore previous instructions"
# inside a claim is itself the claim and must be evaluated, not silently dropped.
# This mirrors the layered defence in ADR-006: we strip the framing, not the content.
# Tag-based role forgery — may span lines, so DOTALL.
_TAG_INJECTION_RE = re.compile(
    r"(?is)"
    r"<\s*system\s*>.*?<\s*/\s*system\s*>"
    r"|<\s*assistant\s*>.*?<\s*/\s*assistant\s*>"
    r"|<\s*user\s*>.*?<\s*/\s*user\s*>",
)
# Single-line ``### SYSTEM:`` framing — must NOT span lines (DOTALL would eat the
# real claim). ``^...$`` with MULTILINE anchors to one line.
_HASH_INJECTION_RE = re.compile(
    r"(?:^###\s*SYSTEM\s*:.*$|^###\s*INSTRUCTION\s*:.*$)",
    re.IGNORECASE | re.MULTILINE,
)
# Collapse runs of whitespace (after stripping markers) for a canonical form.
_WS_RE = re.compile(r"\s+")


class IntakeResult(BaseModel):
    """What ``intake`` yields into the event stream. Tests assert on this shape."""

    accepted: bool
    normalised_text: str
    rejection_reason: str | None = None
    char_count: int = 0


def _strip_injection_markers(text: str) -> str:
    text = _TAG_INJECTION_RE.sub(" ", text)
    text = _HASH_INJECTION_RE.sub(" ", text)
    return text


def _normalise(text: str) -> str:
    # NFKC collapses compatibility chars (e.g. fullwidth Latin) so a model can't
    # be smuggled a "different" A that reads as A.
    text = unicodedata.normalize("NFKC", text)
    text = _strip_injection_markers(text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


async def intake(*, claim_text: str) -> IntakeResult:
    """Normalise and gate a claim. Pure: no ``ctx``, no model, no side effects.

    Bound into the graph via ``FunctionNode(func=intake, parameter_binding='node_input')``
    — the graph passes ``claim_text`` as node input from the user's message.
    """
    raw = claim_text

    if len(raw) > INTAKE_MAX_CHARS:
        result = IntakeResult(
            accepted=False,
            normalised_text="",
            rejection_reason=f"claim exceeds {INTAKE_MAX_CHARS} chars ({len(raw)})",
            char_count=len(raw),
        )
        return result

    normalised = _normalise(raw)
    if not normalised:
        result = IntakeResult(
            accepted=False,
            normalised_text="",
            rejection_reason="claim is empty after normalisation",
            char_count=len(raw),
        )
        return result

    return IntakeResult(
        accepted=True,
        normalised_text=normalised,
        rejection_reason=None,
        char_count=len(raw),
    )


async def write_claim_state(ctx: object, result: IntakeResult) -> None:
    """Persist the intake result into ``ctx.state`` as a :class:`ClaimState`.

    Kept separate from :func:`intake` so the pure logic is unit-testable without an
    ADK context. The graph node wrapper calls both. ``ctx`` is ADK's ``Context``;
    typed as ``object`` here to avoid importing ADK at the boundary (ADR-001).
    """
    state: dict[str, object] = getattr(ctx, "state", {})
    if result.accepted:
        claim = ClaimState(
            raw_text=result.normalised_text,  # store the normalised form; raw is in the event
            normalised_text=result.normalised_text,
            status=ClaimStatus.INTAKED,
        )
    else:
        claim = ClaimState(
            raw_text="",
            normalised_text="",
            status=ClaimStatus.REJECTED,
            rejection_reason=result.rejection_reason,
        )
    state[CLAIM_STATE_KEY] = claim
