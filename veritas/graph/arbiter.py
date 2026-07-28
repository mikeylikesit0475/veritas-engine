"""``arbiter`` node — ARCH §3, node 9.

The sole owner of the verdict narrative. Weighs all fan-out findings (skeptic,
steelman, quant prose) and emits a per-sub-claim verdict with a **calibrated**
confidence. Judgement tier (structured output). ADR-004: schema failures fail
loudly.

The arbiter is the node that makes the confidence loop (2.3) meaningful — it
produces the confidence number that the route uses to decide loop/hitl/report.
ADR-005: the model supplies the confidence; Python decides what to do with it.

The arbiter sees:
- The original claim
- The sub-claims from decompose
- The skeptic/steelman/quant findings for each sub-claim (prose, ADR-004)

It emits a per-sub-claim verdict and an overall claim verdict. The overall
verdict is the combinatorially hardest case (all true → true, any false → false
unless mixed, etc.) — but the arbiter does this by judgement, not by Python rules,
because the interaction between sub-claims can be subtle.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents.context import Context
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.genai import types as genai_types
from pydantic import BaseModel, Field, ValidationError

from veritas.backends_registry import get_backend
from veritas.graph.claim_state import (
    CLAIM_STATE_KEY,
    ClaimState,
    ClaimStatus,
    Verdict,
    VerdictLabel,
)
from veritas.graph.fan_out import FANOUT_RESULT_STATE_KEY, AgentFinding
from veritas.models import Tier, model_for
from veritas.utils import strip_markdown_fences

__all__ = [
    "ARBITER_MODEL_STATE_KEY",
    "CONFIDENCE_THRESHOLD",
    "ArbiterResult",
    "SubClaimVerdict",
    "arbiter",
]

# ARCH §3 node 10: confidence >= tau -> report; < tau and iter < 2 -> loop;
# else -> hitl. This is the threshold the arbiter's confidence is compared
# against. ADR-005: the threshold is in Python, not in the model.
CONFIDENCE_THRESHOLD = 0.7

ARBITER_MODEL_STATE_KEY = "veritas.arbiter.model"

_INSTRUCTION = (
    "You are the arbiter of a fact-checking process. You are given a claim, its "
    "sub-claims, and the findings from two adversarial fact-checkers (a skeptic "
    "who searched for refutation, and a steelman who searched for support) plus "
    "optionally a quant who ran computations.\n\n"
    "Weigh all the findings and emit a JSON object with:\n"
    '- "sub_claim_verdicts": an array of objects, one per sub-claim, each with '
    '"sub_claim_id", "label" (one of "true", "false", "mixed"), "confidence" '
    "(float in [0.0, 1.0]), and " '"rationale" (one sentence citing the key finding).\n'
    '- "overall_verdict": an object with "label", "confidence", and "rationale" '
    "for the claim as a whole.\n\n"
    "The confidence must be CALIBRATED: 0.8 means you'd expect to be right about "
    "80% of the time. Do not inflate confidence. If the evidence is split or "
    'sparse, say 0.5-0.6, not 0.9. Output only the JSON object — no prose, no '
    "markdown fences."
)


class SubClaimVerdict(BaseModel):
    """The arbiter's verdict on one sub-claim."""

    sub_claim_id: str
    label: VerdictLabel = VerdictLabel.UNKNOWN
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    rationale: str = ""


class ArbiterResult(BaseModel):
    """The arbiter's full output — per-sub-claim + overall verdict."""

    sub_claim_verdicts: list[SubClaimVerdict] = Field(default_factory=list)
    overall_verdict: Verdict = Field(default_factory=Verdict)  # type: ignore[arg-type]


def _get_model(ctx: Context) -> BaseLlm:
    model = ctx.state.get(ARBITER_MODEL_STATE_KEY)
    if model is not None:
        return model  # type: ignore[no-any-return]
    app_name = getattr(ctx, "app_name", None) or "veritas"
    try:
        return get_backend(app_name, ARBITER_MODEL_STATE_KEY)  # type: ignore[no-any-return]
    except RuntimeError:
        return model_for(Tier.JUDGEMENT)


def _build_prompt(claim: str, sub_claims: list[dict[str, str]], findings: list[AgentFinding]) -> str:
    """Build the arbiter's prompt from the claim, sub-claims, and findings."""
    sc_text = "\n".join(f"  {sc['id']}: {sc['text']}" for sc in sub_claims)

    findings_by_sc: dict[str, list[AgentFinding]] = {}
    for f in findings:
        findings_by_sc.setdefault(f.sub_claim_id, []).append(f)

    findings_text = ""
    for sc_id, sc_findings in findings_by_sc.items():
        findings_text += f"\n  {sc_id}:\n"
        for f in sc_findings:
            findings_text += f"    [{f.agent_name}] {f.findings}\n"

    return (
        f"{_INSTRUCTION}\n\n"
        f"CLAIM: {claim}\n\n"
        f"SUB-CLAIMS:\n{sc_text}\n\n"
        f"FINDINGS:\n{findings_text}\n\n"
        f"JSON verdict:"
    )


async def _structured_arbiter(
    model: BaseLlm, claim: str, sub_claims: list[dict[str, str]], findings: list[AgentFinding]
) -> ArbiterResult:
    """Ask the judgement tier for a structured arbiter verdict; validate loudly."""
    prompt = _build_prompt(claim, sub_claims, findings)
    contents = [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=prompt)],
        )
    ]
    request = LlmRequest(
        model=model.model,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    raw_text = ""
    async for response in model.generate_content_async(request, stream=False):
        if response.content and response.content.parts:
            for part in response.content.parts:
                if part.text and not part.thought:
                    raw_text += part.text

    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError(
            "arbiter: model returned empty content — expected JSON verdict."
        )

    try:
        return ArbiterResult.model_validate_json(strip_markdown_fences(raw_text))
    except ValidationError as e:
        raise ValueError(
            f"arbiter: model did not return schema-valid JSON. "
            f"Raw: {raw_text[:200]!r}. Error: {e}"
        ) from e


async def arbiter(*, ctx: Context) -> ArbiterResult:
    """Weigh all findings and emit the verdict. Writes back into ``ctx.state``."""
    raw_state: Any = ctx.state
    state_dict: dict[str, object] = (
        raw_state.to_dict() if hasattr(raw_state, "to_dict") else dict(raw_state)
    )
    claim_state = ClaimState.from_state(state_dict)
    claim_text = claim_state.normalised_text or claim_state.raw_text
    if not claim_text:
        raise RuntimeError("arbiter: no claim text in state")

    # Load sub-claims.
    import json

    decompose_data = state_dict.get("veritas.decompose.result")
    if decompose_data is None:
        raise RuntimeError("arbiter: no decompose result in state")
    if isinstance(decompose_data, str):
        decompose_data = json.loads(decompose_data)
    sub_claims: list[dict[str, str]] = decompose_data["sub_claims"]  # type: ignore[index]  # type: ignore[index]

    # Load fan-out findings.
    fanout_data = state_dict.get(FANOUT_RESULT_STATE_KEY)
    if fanout_data is None:
        raise RuntimeError("arbiter: no fan-out result in state")
    if isinstance(fanout_data, str):
        fanout_data = json.loads(fanout_data)
    findings = [AgentFinding.model_validate(f) for f in fanout_data["findings"]]  # type: ignore[index]

    model = _get_model(ctx)
    result = await _structured_arbiter(model, claim_text, sub_claims, findings)

    # Persist the overall verdict into claim state (ARCH §7).
    claim_state.verdict = result.overall_verdict
    claim_state.status = ClaimStatus.ARBITERED
    ctx.state[CLAIM_STATE_KEY] = claim_state

    # Store the full arbiter result for the confidence route (2.3).
    ctx.state["veritas.arbiter.result"] = result.model_dump(mode="json")

    return result
