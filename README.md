# Veritas Engine

> Adversarial fact-checking on Google ADK. **Portfolio proof-of-concept, not a product.**
> The eval table is the deliverable; the multi-agent graph is the packaging. See
> [`ARCHITECTURE.md`](ARCHITECTURE.md) for the contract and [`DECISIONS.md`](DECISIONS.md)
> for the *why*.

## The eval table (the point)

Run `make eval` to reproduce. The benchmark is a curated slice of 40 common-knowledge
claims (true/false/mixed). The single-LLM baseline is one model call per claim — the
cheapest possible fact-checker.

| System | Accuracy | Brier | ECE | Tokens/claim | Wall-clock/claim |
|--------|----------|-------|-----|--------------|-----------------|
| Veritas (full) | _(run `make eval`)_ | _(_)_ | _(_)_ | _(_)_ | _(_)_ |
| Single-LLM baseline | _(_)_ | _(_)_ | _(_)_ | _(_)_ | _(_)_ |

> The numbers are blank because the eval runs live (requires Ollama). `make eval`
> fills them in and renders `docs/reliability-diagram.png`. CI replays recorded
> fixtures — the eval floor is asserted in every build.

**The interview line:** *"Multi-agent debate is a crowded demo. What I can show you
is that it's N points better than single-pass, that its 80%-confidence answers are
right about 80% of the time, and that it costs 6× the tokens — which is why there's a
router in front of it."*

## What works (honest capability list)

| Capability | Status | How to see it |
|-----------|--------|---------------|
| Paste a claim → grounded verdict | ✅ Works | `make sync && python -m veritas.cli "..."` |
| SSE stream of graph events | ✅ Works | `uvicorn veritas.server.app:app` → `http://localhost:8000/ui` |
| Live graph UI (nodes light up) | ✅ Works | Same URL, SVG graph animates during a run |
| Evidence panel (queries + snippets) | ✅ Works | Right panel of the UI |
| Full 13-node adversarial graph | ✅ Works | Triage routes fast/adversarial; decompose → fan-out → arbiter |
| Skeptic + steelman (ADR-003) | ✅ Works | Separate agents, opposed instructions, isolated contexts |
| Confidence loop (bounded at 2) | ✅ Works | Low confidence → loop with new directives → HITL |
| HITL pause + Postgres resume | ✅ Works | `POST /claims/{id}/adjudicate` |
| Citation validator | ✅ Works | Dangling citation → hard fail (test proves it) |
| Prompt-injection defence (ADR-006) | ✅ Works | Stripped + delimited; test proves poisoned page can't flip verdict |
| Docker sandbox (no network) | ✅ Works | Quant agent runs Python in a hardened container |
| Eval: accuracy, Brier, ECE, tokens | ✅ Works | `make eval` prints the table |
| Calibration plot | ✅ Works | `docs/reliability-diagram.png` after `make eval` |
| CI: zero live calls, eval floor | ✅ Works | Recorded fixtures replayed; accuracy drop breaks CI |
| Local model tier (Ollama) | ✅ Works | `VERITAS_FANOUT=ollama` (default) |
| Gemini tier (optional) | ✅ Works | `VERITAS_FANOUT=gemini` (needs key) |

## Quick start

```bash
make sync          # uv sync — install deps
make test-strict   # lint + typecheck + test (the full gate)
make eval          # run the eval (live; requires Ollama) — prints the table
docker compose up  # app + Postgres — the demo
```

All gates must be green. `ruff` + `mypy --strict` are not suggestions.

## Architecture (sketch)

```
  Browser (SVG graph + evidence panel)
      │  SSE: node states, queries, code, verdicts
      │  POST /claims/{id}/adjudicate    (HITL)
      ▼
  ┌──────────────────────────────────────────────────┐
  │ FastAPI service                                  │
  │   ┌────────────────────────────────────────────┐ │
  │   │ ADK graph workflow (13 nodes)              │ │
  │   │   intake → triage →◄route►                │ │
  │   │      fast_check → report                  │ │
  │   │      decompose → ∥{skeptic,steelman,quant}│ │
  │   │                    → arbiter →◄conf route► │ │
  │   │                       loop | hitl | report│ │
  │   └────────────────────────────────────────────┘ │
  │   session service → Postgres  (HITL survives)   │
  │   OTel traces → stdout                          │
  └──────┬───────────────┬──────────────┬────────────┘
         │               │              │
    DuckDuckGo     Code sandbox    Model router
    search         (no network)   ├─ Ollama (default, free)
                                  └─ Gemini (optional, key)
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full contract (13 nodes, safety
invariants, evaluation spec) and [`DECISIONS.md`](DECISIONS.md) for the ADRs that
explain *why*.

## The 13-node graph

| # | Node | What it does |
|---|------|-------------|
| 1 | `intake` | Normalise input, strip injection markers, reject >2000 chars |
| 2 | `triage` | Cost router: fast or adversarial? |
| 3 | route | Python `if` on the triage label (ADR-005) |
| 4 | `fast_check` | One grounded search + verdict (the cheap path) |
| 5 | `decompose` | Claim → 3-5 atomic sub-claims |
| 6 | `skeptic` | Search only for refutation |
| 7 | `steelman` | Search only for support |
| 8 | `quant` | Write + run Python in a sandbox (no network) |
| 9 | `arbiter` | Weigh all findings → verdict + calibrated confidence |
| 10 | route | Python `if` on confidence ≥ τ (ADR-005) |
| 11 | `loop` | Rewrite search directives (bounded at 2 iterations) |
| 12 | `hitl` | Pause; resume via `POST /claims/{id}/adjudicate` |
| 13 | `report` | Markdown with inline citations; dangling citation = hard fail |

## Key decisions (ADRs)

- **ADR-001**: Python + ADK, not a hand-rolled agent loop
- **ADR-003**: Skeptic and steelman are separate agents, not one balanced agent
- **ADR-004**: Fan-out agents never emit structured output
- **ADR-005**: Control flow stays in Python (routing, loop bounds, thresholds)
- **ADR-006**: Fetched web content is untrusted input (strip + delimit + test)
- **ADR-009**: Dual model backend — Ollama default, Gemini optional

Full ADRs in [`DECISIONS.md`](DECISIONS.md).

## Repository layout

```
veritas/
  graph/          13 node definitions + edge wiring
  agents/         skeptic, steelman, quant agent definitions
  tools/          GroundedSearch, FetchPage (typed pydantic args)
  models/         two-tier model factory (Ollama / Gemini)
  sandbox/        Docker sandbox for the quant agent
  server/         FastAPI, SSE, adjudication endpoints
  eval_harness/   record/replay fixtures
evals/
  data/           benchmark slice (40 claims)
  scoring/        accuracy, Brier, ECE, tokens, wall-clock
  fixtures/       recorded backend calls (CI replays these)
docs/             ADRs, interview stories, spike notes
tests/            122 tests (unit, graph, injection, citation, eval, replay)
```

## Testing

| Layer | What | CI |
|-------|------|----|
| Unit | Tools against fake search/fetch | ✅ |
| Graph | Stub model → assert route taken and node visit order | ✅ |
| Injection | Poisoned fetched page → assert verdict unaffected | ✅ |
| Citation | Report with a dangling citation → assert render fails | ✅ |
| Eval | Benchmark slice, recorded fixtures → assert accuracy floor | ✅ |
| Live | One real end-to-end run against Ollama | manual |

CI never calls a live model. Evals replay recorded fixtures from `evals/fixtures/`.