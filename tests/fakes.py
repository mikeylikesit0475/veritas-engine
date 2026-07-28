"""Test helpers: a minimal ``ToolContext`` stand-in for unit-testing tools.

ADK's real :class:`ToolContext` wraps an ``InvocationContext`` and is heavy to
construct. Tools in this repo only touch ``tool_context.state`` (a dict-like mapping),
so a thin fake with a plain dict is sufficient and keeps tests free of ADK internals
(ARCH §9: "Unit | Tools against fake search/fetch | yes").

If a tool ever needs more of the context (e.g. ``tool_context.actions``), extend this
class — don't pull in the real one.
"""

from __future__ import annotations

from typing import Any

__all__ = ["FakeToolContext"]


class FakeToolContext:
    """Just enough ``ToolContext`` for tool unit tests: a ``state`` dict."""

    def __init__(self, state: dict[str, Any] | None = None) -> None:
        self.state: dict[str, Any] = dict(state or {})

    def __repr__(self) -> str:
        return f"FakeToolContext(state={self.state!r})"
