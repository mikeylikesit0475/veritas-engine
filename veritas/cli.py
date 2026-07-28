"""Veritas CLI.

Sprint 0.2 entry point::

    python -m veritas.cli "Is Paris in France?"

Prints the agent's answer to stdout. This is the thinnest possible ADK surface — the
real graph (Sprint 0.5+) replaces ``hello_agent`` with the full workflow.
"""

from __future__ import annotations

import asyncio
import sys

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from veritas.agents import hello_agent

__all__ = ["main"]


async def _run(question: str) -> str:
    """Send one question to the hello agent and return its text answer."""
    session_service = InMemorySessionService()  # type: ignore[no-untyped-call]
    # ^ ADK 2.5's session service ctor is untyped. This is the veritas/graph boundary
    # seam per ADR-001 — upstream drift is contained here, not spread to callers.
    session = await session_service.create_session(app_name="veritas", user_id="cli")
    runner = Runner(
        app_name="veritas",
        agent=hello_agent(),
        session_service=session_service,
    )
    message = genai_types.Content(
        role="user", parts=[genai_types.Part.from_text(text=question)]
    )

    answer = ""
    async for event in runner.run_async(
        user_id="cli", session_id=session.id, new_message=message
    ):
        # Skip user-role echoes and partials; only collect the model's final answer.
        if event.author != "veritas_hello" or event.partial:
            continue
        if event.content and event.content.parts:
            for part in event.content.parts:
                # ``thought=True`` parts are chain-of-thought on reasoning models —
                # never part of the answer the user should see.
                if part.text and not part.thought:
                    answer += part.text

    return answer.strip()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 2 on usage error."""
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print("usage: python -m veritas.cli <question>", file=sys.stderr)
        return 0 if args and args[0] in {"-h", "--help"} else 2

    question = " ".join(args).strip()
    if not question:
        print("error: question must not be empty", file=sys.stderr)
        return 2

    answer = asyncio.run(_run(question))
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
