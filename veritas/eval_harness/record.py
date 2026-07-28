"""Record a fixture from a live run.

Usage::

    # Record: runs the graph against live backends, saves the fixture.
    VERITAS_RECORD=1 python -m veritas.eval_harness.record "The Eiffel Tower is in Berlin." --name eiffel_berlin

    # The fixture is saved to evals/fixtures/<name>.json.

This is a local-only tool. CI never runs it (CLAUDE.md #6). It exists so that
fixtures are recorded before the swarm gets expensive (Sprint 1.7 / Sprint 3).
"""

from __future__ import annotations

import argparse
import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from veritas.backends_registry import clear_backends, register_backends
from veritas.eval_harness.backends import RecordingModel, RecordingSearchBackend
from veritas.eval_harness.fixtures import Fixture, save_fixture
from veritas.graph.claim_state import ClaimState
from veritas.graph.fast_check import FAST_CHECK_MODEL_STATE_KEY
from veritas.graph.wiring import build_graph
from veritas.models import Tier, model_for
from veritas.server.backends import HttpSearchBackend
from veritas.tools.grounded_search import SEARCH_BACKEND_STATE_KEY


async def _record(claim: str, name: str) -> str:
    """Run the graph live, recording all backend calls. Returns the fixture path."""
    fixture = Fixture(name=name, claim=claim)

    real_search = HttpSearchBackend()
    real_model = model_for(Tier.JUDGEMENT)

    recording_search = RecordingSearchBackend(real_search, fixture)
    recording_model = RecordingModel(real_model, fixture)

    register_backends(
        "veritas",
        **{
            SEARCH_BACKEND_STATE_KEY: recording_search,
            FAST_CHECK_MODEL_STATE_KEY: recording_model,
        },
    )

    try:
        session_service = InMemorySessionService()  # type: ignore[no-untyped-call]
        session = await session_service.create_session(
            app_name="veritas", user_id="record"
        )
        runner = Runner(
            app_name="veritas",
            agent=build_graph(),  # type: ignore[arg-type]
            session_service=session_service,
        )

        message = genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=claim)],
        )

        async for _event in runner.run_async(
            user_id="record", session_id=session.id, new_message=message
        ):
            pass  # consume all events; the recording backends capture the data

        # Extract the final verdict from session state.
        final = await session_service.get_session(
            app_name="veritas", user_id="record", session_id=session.id
        )
        if final is not None:
            raw = final.state
            state_dict = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
            claim_state = ClaimState.from_state(state_dict)
            fixture.verdict = claim_state.verdict

    finally:
        await real_search.aclose()
        clear_backends("veritas")

    path = save_fixture(fixture)
    print(f"Fixture saved: {path}")
    print(f"  search calls: {len(fixture.search_calls)}")
    print(f"  model calls:  {len(fixture.model_calls)}")
    print(f"  verdict:      {fixture.verdict.label} (conf={fixture.verdict.confidence})")
    return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record an eval fixture from a live run.")
    parser.add_argument("claim", help="The claim to check.")
    parser.add_argument("--name", required=True, help="Fixture name (filename in evals/fixtures/).")
    args = parser.parse_args(argv)

    asyncio.run(_record(args.claim, args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
