# Veritas Engine

> Adversarial fact-checking on Google ADK. **Portfolio proof-of-concept, not a product.**
> The eval table is the deliverable; the multi-agent graph is the packaging. See
> [`ARCHITECTURE.md`](ARCHITECTURE.md) for the contract and [`DECISIONS.md`](DECISIONS.md)
> for the *why*.

## Status

Sprint 0 in progress — skeleton. No claims are checked yet. This README stays honest about
what works: right now, the toolchain and an empty test suite. See [`BACKLOG.md`](BACKLOG.md)
for the plan.

## Quick start

```bash
make sync      # uv sync — install dev deps
make lint      # ruff check
make typecheck # mypy --strict
make test      # pytest (no live model calls)
```

All gates must be green. `ruff` + `mypy --strict` are not suggestions.

## Repository layout

```
veritas/            the package (empty — Sprint 0.2+ fills it)
tests/              unit / graph / injection / citation / eval tests
evals/              benchmark slices + recorded fixtures (Sprint 1+)
docs/               ADRs (in DECISIONS.md), interview stories, scope cuts
```

See [`ARCHITECTURE.md` §8](ARCHITECTURE.md) for the target layout as the sprints land.