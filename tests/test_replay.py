"""CI replay test — asserts a fixture replays to the recorded verdict (ARCH §6).

This is the test that mechanically enforces "CI never calls a live model"
(CLAUDE.md #6). It loads a recorded fixture, runs the graph with replay backends,
and asserts the verdict matches. No network, no model, no API key.

If a fixture is missing or stale, the replay backends raise — that's the guarantee.
"""

from __future__ import annotations

import pytest
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from veritas.backends_registry import clear_backends, register_backends
from veritas.eval_harness.backends import ReplayModel, ReplaySearchBackend
from veritas.eval_harness.fixtures import load_fixture
from veritas.graph.arbiter import ARBITER_MODEL_STATE_KEY
from veritas.graph.claim_state import ClaimState, ClaimStatus
from veritas.graph.decompose import DECOMPOSE_MODEL_STATE_KEY
from veritas.graph.fast_check import FAST_CHECK_MODEL_STATE_KEY
from veritas.graph.triage import TRIAGE_MODEL_STATE_KEY
from veritas.graph.wiring import build_graph
from veritas.tools.grounded_search import SEARCH_BACKEND_STATE_KEY

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.eval,
]

# The fixtures this test replays. Each must exist in evals/fixtures/.
# Sprint 0.6 has one; Sprint 1.7 adds more before the swarm gets expensive.
FIXTURE_NAMES = ["eiffel_berlin", "great_wall_space"]


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
async def test_fixture_replays_to_recorded_verdict(fixture_name: str) -> None:
    # Load the fixture. If it doesn't exist, this test fails loudly — the fixture
    # must be recorded before CI runs (Sprint 1.7 / record mode).
    fixture = load_fixture(fixture_name)

    # Wire replay backends into the registry. Zero live calls.
    register_backends(
        "veritas",
        **{
            SEARCH_BACKEND_STATE_KEY: ReplaySearchBackend(fixture),
            FAST_CHECK_MODEL_STATE_KEY: ReplayModel(fixture),
            TRIAGE_MODEL_STATE_KEY: ReplayModel(fixture),
            DECOMPOSE_MODEL_STATE_KEY: ReplayModel(fixture),
            ARBITER_MODEL_STATE_KEY: ReplayModel(fixture),
        },
    )

    try:
        session_service = InMemorySessionService()  # type: ignore[no-untyped-call]
        session = await session_service.create_session(  # type: ignore[no-any-return]
            app_name="veritas", user_id="replay"
        )
        runner = Runner(
            app_name="veritas",
            agent=build_graph(),  # type: ignore[arg-type]
            session_service=session_service,
        )

        message = genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=fixture.claim)],
        )

        async for _event in runner.run_async(
            user_id="replay", session_id=session.id, new_message=message
        ):
            pass

        # Assert the verdict matches the recorded one.
        final = await session_service.get_session(
            app_name="veritas", user_id="replay", session_id=session.id
        )
        assert final is not None, "session vanished after run"
        raw_state = final.state
        state_dict = raw_state.to_dict() if hasattr(raw_state, "to_dict") else dict(raw_state)  # type: ignore[union-attr]
        claim_state = ClaimState.from_state(state_dict)

        assert claim_state.status is ClaimStatus.REPORTED
        assert claim_state.verdict.label == fixture.verdict.label
        assert claim_state.verdict.confidence == pytest.approx(
            fixture.verdict.confidence, abs=0.01
        )
    finally:
        clear_backends("veritas")


async def test_replay_raises_on_unrecorded_query() -> None:
    # The guarantee: if a query isn't in the fixture, replay raises rather than
    # silently calling a live backend. This is the invariant that makes CI honest.
    fixture = load_fixture("eiffel_berlin")
    backend = ReplaySearchBackend(fixture)

    with pytest.raises(KeyError, match="not in fixture"):
        await backend.search("this query was never recorded")


async def test_replay_raises_on_unrecorded_prompt() -> None:
    fixture = load_fixture("eiffel_berlin")
    model = ReplayModel(fixture)

    from google.adk.models.llm_request import LlmRequest

    request = LlmRequest(
        model="veritas-replay",
        contents=[
            genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text="a prompt that was never recorded")],
            )
        ],
    )

    with pytest.raises(KeyError, match="not in fixture"):
        async for _ in model.generate_content_async(request):
            pass
