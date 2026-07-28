"""CI eval floor test — ARCH §6, Sprint 3.6.

> Lowering accuracy breaks CI.

This test asserts a *floor* on the eval results. If the system's accuracy drops
below the floor, CI fails — that's the regression gate.

In CI, we use the recorded fixtures (no live calls, CLAUDE.md #6). The floor is
set conservatively: the PoC's curated benchmark is small (40 claims), so a single
failure changes accuracy by 2.5%. The floor reflects "the system works on the
simple cases it was designed for," not "the system is production-ready."

When run live (``make eval``), the full 40-claim eval produces the actual numbers.
CI replays the 2 recorded fixtures and asserts the floor on those.
"""

from __future__ import annotations

import pytest

from veritas.eval_harness.fixtures import load_fixture

pytestmark = [pytest.mark.eval]

# The floor: the system must get at least this many of the recorded fixtures
# correct. Currently 2 fixtures, both "false" — the floor is 2/2 = 100%.
# When more fixtures are added, adjust this floor proportionally.
ACCURACY_FLOOR = 1.0  # 100% of recorded fixtures


@pytest.mark.parametrize(
    "fixture_name",
    ["eiffel_berlin", "great_wall_space"],
)
@pytest.mark.asyncio
async def test_recorded_fixture_meets_accuracy_floor(fixture_name: str) -> None:
    """Each recorded fixture must produce the correct verdict (the floor)."""
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types as genai_types

    from veritas.backends_registry import clear_backends, register_backends
    from veritas.eval_harness.backends import ReplayModel, ReplaySearchBackend
    from veritas.graph.arbiter import ARBITER_MODEL_STATE_KEY
    from veritas.graph.claim_state import ClaimState
    from veritas.graph.decompose import DECOMPOSE_MODEL_STATE_KEY
    from veritas.graph.fast_check import FAST_CHECK_MODEL_STATE_KEY
    from veritas.graph.triage import TRIAGE_MODEL_STATE_KEY
    from veritas.graph.wiring import build_graph
    from veritas.tools.grounded_search import SEARCH_BACKEND_STATE_KEY

    fixture = load_fixture(fixture_name)
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
            app_name="veritas", user_id="eval-floor"
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

        async for _ in runner.run_async(
            user_id="eval-floor", session_id=session.id, new_message=message
        ):
            pass

        final = await session_service.get_session(
            app_name="veritas", user_id="eval-floor", session_id=session.id
        )
        assert final is not None, "session vanished"
        raw = final.state
        state_dict = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
        claim = ClaimState.from_state(state_dict)

        # The floor: the verdict must match the recorded verdict.
        assert claim.verdict.label == fixture.verdict.label, (
            f"Fixture {fixture_name}: verdict {claim.verdict.label} does not match "
            f"recorded {fixture.verdict.label}. The accuracy floor is broken."
        )
    finally:
        clear_backends("veritas")


def test_accuracy_floor_is_set() -> None:
    """The floor must be a positive number. Documenting it here makes a change
    visible in a diff."""
    assert ACCURACY_FLOOR > 0.0
    assert ACCURACY_FLOOR <= 1.0
