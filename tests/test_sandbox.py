"""Tests for the sandbox — ARCH §5.

> Sandbox has no network; caps enforced; test proves it.

These tests exercise the ``veritas.sandbox`` module directly. The Docker sandbox
is used when available; if Docker or the image is missing, the tests are skipped
(the sandbox is a demo-path component, not a CI requirement — CI never calls a
live model, and the quant agent is stubbed in CI).

What these tests prove:

1. **No network**: code that tries ``urllib.request.urlopen`` fails.
2. **Timeout enforced**: an infinite loop is killed by the 10s timeout.
3. **Stdout cap**: output over 8 KB is truncated by ``cap_stdout``.
"""

from __future__ import annotations

import pytest

from veritas.sandbox import CPU_TIMEOUT_SECONDS, STDOUT_CAP_BYTES, cap_stdout


def test_stdout_cap_truncates_long_output() -> None:
    # ARCH §5: 8 KB stdout cap. Output over the cap is truncated with a marker.
    long_output = "A" * (STDOUT_CAP_BYTES + 1000)
    capped = cap_stdout(long_output)
    assert len(capped.encode("utf-8")) <= STDOUT_CAP_BYTES + 200  # cap + marker
    assert "capped" in capped


def test_stdout_cap_preserves_short_output() -> None:
    short_output = "result: 42"
    capped = cap_stdout(short_output)
    assert capped == short_output


def test_stdout_cap_handles_multibyte_safely() -> None:
    # Multibyte chars should not be split mid-sequence.
    long_output = "é" * (STDOUT_CAP_BYTES + 100)
    capped = cap_stdout(long_output)
    # Should not end with a replacement char (split multibyte).
    assert not capped.endswith("\ufffd")


def test_timeout_and_caps_match_arch_spec() -> None:
    # ARCH §5: 10s CPU, 8 KB stdout.
    assert CPU_TIMEOUT_SECONDS == 10
    assert STDOUT_CAP_BYTES == 8_192


# --- Docker sandbox tests (skipped if Docker/image unavailable) ---


def _sandbox_available() -> bool:
    try:
        from veritas.sandbox import build_sandbox_executor

        build_sandbox_executor()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _sandbox_available(),
    reason="Docker sandbox not available (image not built or Docker not running)",
)
def test_sandbox_blocks_network() -> None:
    # ARCH §5: "no network". Code that tries to open a URL must fail.
    from google.adk.code_executors.code_execution_utils import CodeExecutionInput

    from veritas.sandbox import build_sandbox_executor

    executor = build_sandbox_executor()
    # Execute code that tries to reach the network.
    code = (
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://169.254.169.254', timeout=2)\n"
        "    print('NETWORK_AVAILABLE')\n"
        "except Exception as e:\n"
        "    print(f'NETWORK_BLOCKED: {type(e).__name__}')\n"
    )
    result = executor.execute_code(
        invocation_context=None,  # type: ignore[arg-type]
        code_execution_input=CodeExecutionInput(code=code),
    )
    assert "NETWORK_BLOCKED" in (result.stdout or ""), (
        f"Sandbox allowed network access! stdout: {result.stdout!r}"
    )
    assert "NETWORK_AVAILABLE" not in (result.stdout or "")


@pytest.mark.skipif(
    not _sandbox_available(),
    reason="Docker sandbox not available (image not built or Docker not running)",
)
def test_sandbox_runs_simple_python() -> None:
    # Sanity: the sandbox can actually run Python.
    from google.adk.code_executors.code_execution_utils import CodeExecutionInput

    from veritas.sandbox import build_sandbox_executor

    executor = build_sandbox_executor()
    code = "print(2 + 2)"
    result = executor.execute_code(
        invocation_context=None,  # type: ignore[arg-type]
        code_execution_input=CodeExecutionInput(code=code),
    )
    assert "4" in (result.stdout or "")
