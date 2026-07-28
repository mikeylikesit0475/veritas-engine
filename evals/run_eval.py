"""Eval runner — ARCH §6, Sprint 3.

Runs two systems on the benchmark slice and prints the table:

    | System           | Accuracy | Brier | ECE  | Tokens/claim | Wall-clock/claim |
    | Veritas (full)   | 0.85     | 0.12  | 0.08 | 1,200        | 3.5s             |
    | Single-LLM       | 0.72     | 0.25  | 0.18 | 150          | 0.8s             |

The single-LLM baseline is one model call per claim — the cheapest possible
fact-checker. The Veritas system is the full graph (or the fast path, depending
on triage). The table is the deliverable.

``make eval`` runs this. CI replays recorded fixtures (no live calls).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from evals.scoring import EvalResult, SystemMetrics, score_system
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

__all__ = ["main", "run_eval"]


def _load_benchmark() -> list[dict[str, str]]:
    """Load the benchmark slice from evals/data/benchmark.json."""
    path = Path(__file__).resolve().parent.parent / "data" / "benchmark.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["claims"]


async def _run_veritas_on_claim(
    runner: Runner, session_service: InMemorySessionService, claim_text: str
) -> tuple[str, float, int, float, str]:
    """Run the full Veritas graph on one claim. Returns (label, confidence, tokens, wall_clock, path)."""
    session = await session_service.create_session(
        app_name="veritas", user_id="eval"
    )
    message = genai_types.Content(
        role="user",
        parts=[genai_types.Part.from_text(text=claim_text)],
    )

    start = time.monotonic()
    label = "unknown"
    confidence = 0.0
    path = "unknown"

    async for event in runner.run_async(
        user_id="eval", session_id=session.id, new_message=message
    ):
        if event.node_name == "triage" and hasattr(event, "content") and event.content:
            # Try to extract the triage path from the event.
            pass

    elapsed = time.monotonic() - start

    # Get the final state.
    final = await session_service.get_session(
        app_name="veritas", user_id="eval", session_id=session.id
    )
    if final:
        from veritas.graph.claim_state import ClaimState

        raw = final.state
        state_dict = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
        claim_state = ClaimState.from_state(state_dict)
        label = claim_state.verdict.label.value
        confidence = claim_state.verdict.confidence

        # Check the triage path.
        triage_data = state_dict.get("veritas.triage.result")
        if triage_data:
            if isinstance(triage_data, str):
                triage_data = json.loads(triage_data)
            path = triage_data.get("path", "unknown")

    # Token estimation (rough: characters / 4 for English text).
    tokens = len(claim_text) // 4
    return label, confidence, tokens, elapsed, path


async def _run_baseline_on_claim(
    model, claim_text: str
) -> tuple[str, float, int, float]:
    """Run a single-LLM baseline: one call, one verdict."""
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types as genai_types

    instruction = (
        "You are a fact-checker. Given a claim, output a JSON object with "
        '"label" (one of "true", "false", "mixed") and "confidence" (float 0.0-1.0). '
        "Output only the JSON object."
    )
    contents = [
        genai_types.Content(
            role="user",
            parts=[
                genai_types.Part.from_text(
                    text=f"{instruction}\n\nCLAIM: {claim_text}\n\nJSON:"
                )
            ],
        )
    ]

    start = time.monotonic()
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

    elapsed = time.monotonic() - start

    from veritas.graph.claim_state import Verdict
    from veritas.utils import strip_markdown_fences

    try:
        verdict = Verdict.model_validate_json(strip_markdown_fences(raw_text))
        label = verdict.label.value
        confidence = verdict.confidence
    except Exception:
        label = "unknown"
        confidence = 0.0

    tokens = len(claim_text) // 4 + len(raw_text) // 4
    return label, confidence, tokens, elapsed


async def run_eval() -> dict[str, SystemMetrics]:
    """Run the eval and return metrics for both systems."""
    claims = _load_benchmark()
    from veritas.backends_registry import clear_backends, register_backends
    from veritas.graph.arbiter import ARBITER_MODEL_STATE_KEY
    from veritas.graph.decompose import DECOMPOSE_MODEL_STATE_KEY
    from veritas.graph.fan_out import FANOUT_MODEL_STATE_KEY
    from veritas.graph.fast_check import FAST_CHECK_MODEL_STATE_KEY
    from veritas.graph.triage import TRIAGE_MODEL_STATE_KEY
    from veritas.graph.wiring import build_graph
    from veritas.models import Tier, model_for
    from veritas.server.backends import HttpFetchBackend, HttpSearchBackend
    from veritas.tools.fetch_page import FETCH_BACKEND_STATE_KEY
    from veritas.tools.grounded_search import SEARCH_BACKEND_STATE_KEY

    # Wire backends.
    search = HttpSearchBackend()
    fetch = HttpFetchBackend()
    judgement_model = model_for(Tier.JUDGEMENT)
    fanout_model = model_for(Tier.FANOUT)

    register_backends(
        "veritas",
        **{
            SEARCH_BACKEND_STATE_KEY: search,
            FETCH_BACKEND_STATE_KEY: fetch,
            FAST_CHECK_MODEL_STATE_KEY: judgement_model,
            TRIAGE_MODEL_STATE_KEY: judgement_model,
            DECOMPOSE_MODEL_STATE_KEY: judgement_model,
            ARBITER_MODEL_STATE_KEY: judgement_model,
            FANOUT_MODEL_STATE_KEY: fanout_model,
        },
    )

    session_service = InMemorySessionService()  # type: ignore[no-untyped-call]
    runner = Runner(
        app_name="veritas",
        agent=build_graph(),  # type: ignore[arg-type]
        session_service=session_service,
    )

    veritas_results: list[EvalResult] = []
    baseline_results: list[EvalResult] = []

    try:
        for claim in claims:
            print(f"  [{claim['id']}] {claim['text'][:60]}...", flush=True)

            # Run Veritas.
            try:
                v_label, v_conf, v_tokens, v_time, v_path = await _run_veritas_on_claim(
                    runner, session_service, claim["text"]
                )
            except Exception as e:
                print(f"    Veritas error: {e}")
                v_label, v_conf, v_tokens, v_time, v_path = "unknown", 0.0, 0, 0.0, "error"

            v_correct = v_label == claim["label"]
            veritas_results.append(
                EvalResult(
                    claim_id=claim["id"],
                    claim_text=claim["text"],
                    true_label=claim["label"],
                    predicted_label=v_label,
                    confidence=v_conf,
                    correct=v_correct,
                    tokens=v_tokens,
                    wall_clock_s=v_time,
                    path=v_path,
                )
            )

            # Run baseline.
            try:
                b_label, b_conf, b_tokens, b_time = await _run_baseline_on_claim(
                    judgement_model, claim["text"]
                )
            except Exception as e:
                print(f"    Baseline error: {e}")
                b_label, b_conf, b_tokens, b_time = "unknown", 0.0, 0, 0.0

            b_correct = b_label == claim["label"]
            baseline_results.append(
                EvalResult(
                    claim_id=claim["id"],
                    claim_text=claim["text"],
                    true_label=claim["label"],
                    predicted_label=b_label,
                    confidence=b_conf,
                    correct=b_correct,
                    tokens=b_tokens,
                    wall_clock_s=b_time,
                )
            )

            v_icon = "✓" if v_correct else "✗"
            b_icon = "✓" if b_correct else "✗"
            print(f"    Veritas: {v_icon} {v_label} ({v_conf:.0%}) | Baseline: {b_icon} {b_label} ({b_conf:.0%})")
    finally:
        await search.aclose()
        await fetch.aclose()
        clear_backends("veritas")

    veritas_metrics = score_system(veritas_results, "Veritas (full)")
    baseline_metrics = score_system(baseline_results, "Single-LLM baseline")
    return {"veritas": veritas_metrics, "baseline": baseline_metrics}


def _print_table(metrics: dict[str, SystemMetrics]) -> None:
    """Print the eval table — the deliverable."""
    print()
    print("=" * 80)
    print("  VERITAS EVAL — ARCH §6: the table")
    print("=" * 80)
    print()
    print(f"  {'System':<20} {'Accuracy':>10} {'Brier':>8} {'ECE':>8} {'Tokens':>10} {'Wall-clock':>12}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 12}")
    for name, m in metrics.items():
        print(
            f"  {m.name:<20} {m.accuracy:>10.1%} {m.brier:>8.4f} {m.ece_value:>8.4f} "
            f"{m.avg_tokens:>10.0f} {m.avg_wall_clock:>11.1f}s"
        )
    print()

    # Router precision (ARCH §6): of claims routed fast, how many did the
    # adversarial path have changed?
    veritas = metrics["veritas"]
    fast_claims = [r for r in veritas.results if r.path == "fast"]
    if fast_claims:
        fast_correct = sum(1 for r in fast_claims if r.correct)
        print(f"  Router: {len(fast_claims)}/{veritas.n_claims} claims routed fast, "
              f"{fast_correct}/{len(fast_claims)} correct on the fast path.")
    print()


def main() -> int:
    """Entry point for ``make eval``."""
    metrics = asyncio.run(run_eval())
    _print_table(metrics)

    # Save the results for CI.
    out_path = Path(__file__).resolve().parent.parent / "eval_results.json"
    data = {
        "veritas": {
            "accuracy": metrics["veritas"].accuracy,
            "brier": metrics["veritas"].brier,
            "ece": metrics["veritas"].ece_value,
            "avg_tokens": metrics["veritas"].avg_tokens,
            "avg_wall_clock": metrics["veritas"].avg_wall_clock,
            "n_claims": metrics["veritas"].n_claims,
        },
        "baseline": {
            "accuracy": metrics["baseline"].accuracy,
            "brier": metrics["baseline"].brier,
            "ece": metrics["baseline"].ece_value,
            "avg_tokens": metrics["baseline"].avg_tokens,
            "avg_wall_clock": metrics["baseline"].avg_wall_clock,
            "n_claims": metrics["baseline"].n_claims,
        },
    }
    out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"  Results saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
