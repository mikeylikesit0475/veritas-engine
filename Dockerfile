FROM python:3.11-slim AS base

# Veritas Engine — portfolio PoC.
# Dockerfile for the app service. docker compose builds this + Postgres.

WORKDIR /app

# System deps: build tools for uv, ca-certificates for HTTPS, curl for healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (faster than pip for dependency resolution).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy only dependency manifests first (cache layer).
COPY pyproject.toml uv.lock ./

# Install dependencies (no live model calls at build time).
RUN uv sync --frozen --no-dev

# Copy the project.
COPY veritas/ veritas/
COPY evals/ evals/
COPY tests/ tests/
COPY Makefile ARCHITECTURE.md DECISIONS.md DEMO_SCRIPT.md README.md ./

# The server runs on port 8000. Postgres for sessions.
EXPOSE 8000

# Health check: the server exposes /health.
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server. VERITAS_DB_URL points to the compose Postgres service.
ENV VERITAS_DB_URL=postgresql+asyncpg://veritas:veritas@postgres:5432/veritas
ENV VERITAS_FANOUT=ollama
ENV VERITAS_JUDGEMENT=ollama

CMD ["uv", "run", "uvicorn", "veritas.server.app:app", "--host", "0.0.0.0", "--port", "8000"]