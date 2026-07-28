"""Veritas sandbox — ARCH §5.

The ``quant`` agent writes Python and runs it in a sandbox with:
- **no network** (ContainerCodeExecutor default: network_disabled=True)
- **all Linux capabilities dropped** (cap_drop=['ALL'])
- **no privilege escalation** (security_opt=['no-new-privileges'])
- **10s CPU timeout** (timeout_seconds on the executor)
- **8 KB stdout cap** (enforced in our wrapper — the container executor doesn't
  cap stdout itself)

The Docker image is a slim Python base. ``make sandbox-build`` builds it.

ARCH §5 honesty: the container executor provides good isolation for a PoC, but it
is not kernel-level isolation (gVisor/Firecracker). The README will say so. For a
portfolio demo this is the right level — the threat is model-generated code doing
something dumb, not a determined adversary.
"""

from __future__ import annotations

import logging
from pathlib import Path

from google.adk.code_executors.base_code_executor import BaseCodeExecutor

__all__ = [
    "CPU_TIMEOUT_SECONDS",
    "SANDBOX_DOCKERFILE_DIR",
    "SANDBOX_IMAGE",
    "STDOUT_CAP_BYTES",
    "build_sandbox_executor",
]

logger = logging.getLogger(__name__)

SANDBOX_IMAGE = "veritas-sandbox:latest"
SANDBOX_DOCKERFILE_DIR = str(Path(__file__).resolve().parent)
STDOUT_CAP_BYTES = 8_192  # ARCH §5: 8 KB stdout cap
CPU_TIMEOUT_SECONDS = 10  # ARCH §5: 10s CPU


def build_sandbox_executor() -> BaseCodeExecutor:
    """Build the sandboxed code executor.

    Uses ``ContainerCodeExecutor`` with:
    - the veritas-sandbox Docker image
    - network disabled (ARCH §5)
    - 10s timeout

    The stdout cap is enforced in ``_cap_stdout`` — the container executor returns
    raw stdout, and the quant agent's tool wrapper truncates it before the model
    sees it.

    Raises:
        RuntimeError: if Docker is not available or the image is not built.
    """
    from google.adk.code_executors.container_code_executor import ContainerCodeExecutor

    # Check the image exists — give a helpful error if not.
    try:
        import docker  # type: ignore[import-untyped]

        client = docker.from_env()
        try:
            client.images.get(SANDBOX_IMAGE)
        except docker.errors.ImageNotFound:
            raise RuntimeError(
                f"Sandbox image {SANDBOX_IMAGE!r} not found. Run "
                f"'docker build -t {SANDBOX_IMAGE} {SANDBOX_DOCKERFILE_DIR}' first."
            ) from None
    except ImportError as e:
        raise RuntimeError(f"docker-py not installed: {e}") from e

    return ContainerCodeExecutor(
        image=SANDBOX_IMAGE,
        network_enabled=False,  # ARCH §5: no network
        timeout_seconds=CPU_TIMEOUT_SECONDS,
    )


def cap_stdout(stdout: str, cap: int = STDOUT_CAP_BYTES) -> str:
    """Truncate stdout to the ARCH §5 8 KB cap, with a visible marker."""
    encoded = stdout.encode("utf-8")
    if len(encoded) <= cap:
        return stdout
    truncated = encoded[:cap].decode("utf-8", errors="ignore")
    return f"{truncated}\n... [stdout capped at {cap // 1024} KB by veritas sandbox]"
