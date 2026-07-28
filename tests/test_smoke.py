"""Smoke test for the Sprint 0.1 skeleton.

Asserts only what Sprint 0.1 promises: the package imports and reports a version.
This is the floor — Sprint 0.2 replaces it with the first graph test.
"""

import veritas


def test_package_imports() -> None:
    assert veritas.__name__ == "veritas"


def test_version_is_declared() -> None:
    assert isinstance(veritas.__version__, str)
    assert veritas.__version__ == "0.0.0"
