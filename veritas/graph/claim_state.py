"""Claim state — the projection of the event stream (ARCH §7).

One source of truth. The graph nodes read from and write to ``ctx.state``; this
module defines the typed shape of the keys they care about. ADK's ``State`` is a
dict-like mapping, so we store the serialised :class:`ClaimState` under
:data:`CLAIM_STATE_KEY` and re-validate on read. This keeps the "no dict[str, Any]
past the boundary" convention (CLAUDE.md) honest without fighting ADK's state model.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

__all__ = [
    "CLAIM_STATE_KEY",
    "ClaimState",
    "ClaimStatus",
    "Verdict",
    "VerdictLabel",
]

CLAIM_STATE_KEY = "veritas.claim.state"


class ClaimStatus(StrEnum):
    """Where the claim is in the workflow. Mirrors ARCH §3 node names."""

    INTAKED = "intaked"
    TRIAGED = "triaged"
    FAST_CHECKED = "fast_checked"
    DECOMPOSED = "decomposed"
    FAN_OUT_DONE = "fan_out_done"
    ARBITERED = "arbitered"
    LOOPED = "looped"
    HITL_PAUSED = "hitl_paused"
    REPORTED = "reported"
    REJECTED = "rejected"  # intake refused the claim (too long / injection markers)


class VerdictLabel(StrEnum):
    """The three verdicts the system can emit. Plus ``UNKNOWN`` pre-arbiter."""

    TRUE = "true"
    FALSE = "false"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class Verdict(BaseModel):
    """A verdict on the claim (or a sub-claim, in later sprints)."""

    label: VerdictLabel = VerdictLabel.UNKNOWN
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    rationale: str = ""


class ClaimState(BaseModel):
    """The whole claim, as a projection of the event stream so far."""

    raw_text: str = ""
    normalised_text: str = ""
    status: ClaimStatus = ClaimStatus.INTAKED
    rejection_reason: str | None = None
    # Populated by fast_check (Sprint 0.5) and arbiter (Sprint 2).
    verdict: Verdict = Field(default_factory=Verdict)  # type: ignore[arg-type]
    # Sub-claims and per-sub-claim evidence land in Sprint 1.
    sub_claims: list[str] = Field(default_factory=list)

    def to_state_dict(self) -> dict[str, object]:
        """Serialise into the dict ADK's ``State`` expects."""
        return {CLAIM_STATE_KEY: self.model_dump(mode="json")}

    @classmethod
    def from_state(cls, state: dict[str, object]) -> ClaimState:
        """Re-validate from a ``State`` dict. Absent key -> fresh claim."""
        raw = state.get(CLAIM_STATE_KEY)
        if raw is None:
            return cls()
        if isinstance(raw, ClaimState):
            return raw
        return cls.model_validate(raw)
