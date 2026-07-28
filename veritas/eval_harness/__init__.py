"""Eval harness — ARCH §6, the deliverable.

Sprint 0.6 ships the record/replay infrastructure that makes CI free and offline:

- ``record`` mode (``VERITAS_RECORD=1``): wrapping backends record live calls to
  ``evals/fixtures/<name>/``. Run locally with real backends.
- ``replay`` mode (default in CI): replay backends serve recorded responses. Zero
  network calls. CI asserts the verdict matches the recorded one.

Sprint 3 adds the benchmark slice, scoring, and the eval table.
"""

from __future__ import annotations

from veritas.eval_harness.backends import (
    RecordingModel,
    RecordingSearchBackend,
    ReplayModel,
    ReplaySearchBackend,
)
from veritas.eval_harness.fixtures import Fixture, load_fixture

__all__ = [
    "Fixture",
    "RecordingModel",
    "RecordingSearchBackend",
    "ReplayModel",
    "ReplaySearchBackend",
    "load_fixture",
]
