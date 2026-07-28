# Veritas Engine — Backlog & Sprint Plan

Scrum master's file. One-week sprints. Timeboxes are hours of focused work — **if an item
blows its box it gets cut or split, not extended.** Raise it, don't absorb it.

## Definition of done (every sprint)

- [ ] Working demo of the sprint goal, start to finish, without editing code
- [ ] Tests for what was added, green in CI
- [ ] `DEMO_SCRIPT.md` updated to match reality
- [ ] One interview story in `docs/interview-stories/`

## Definition of done (whole repo)

README with the **eval table first**, architecture sketch, demo GIF · automated tests ·
GitHub Actions CI · Dockerfile · 2-minute demo performable live.

---

## Sprint 0 — Skeleton · goal: *"one claim goes in, one grounded verdict comes out"*

No swarm. Eliminate environment and ADK-surface risk before the architecture depends on it.

| # | Item | Box | Done when |
|---|------|-----|-----------|
| 0.1 | Repo init, `uv`/`pyproject`, ruff + mypy, GitHub Actions CI, push | 2h | Green CI on an empty suite |
| 0.2 | `google-adk` pinned; hello-world agent answers one question via Gemini | 2h | `python -m veritas.cli "..."` prints an answer |
| 0.3 | **Spike:** does ADK Python ship a Postgres-backed session service with HITL resume? | 3h | Written answer in `docs/adk-notes.md`; if no, a plan sized |
| 0.4 | `GroundedSearch` + `FetchPage` tools, typed pydantic args, guards from ARCH §5 | 4h | Unit tests against fakes pass |
| 0.5 | `intake` + `fast_check` nodes; FastAPI `POST /claims`; SSE stream | 4h | Browser shows a streamed verdict for a simple claim |
| 0.6 | Stub-model harness: replay fixtures, zero live calls in CI | 3h | `make test` is green with no API key set |

**Demo:** paste "The Eiffel Tower is in Berlin" → grounded refutation, streamed, with the
search query visible. **Total ~18h.**

---

## Sprint 1 — Decompose & fan out · goal: *"the agent shows its work"*

| # | Item | Box | Done when |
|---|------|-----|-----------|
| 1.1 | `decompose` node, structured output → 3–5 atomic sub-claims | 4h | A compound claim splits sensibly; schema-validated |
| 1.2 | `skeptic` + `steelman` agents in **Task** mode, separate instructions | 5h | Each returns findings; contexts are isolated |
| 1.3 | Parallel fan-out across sub-claims × 2 agents; concurrency capped | 4h | 8 agents run concurrently; timings logged |
| 1.4 | Two-tier model factory; fan-out on the local backend | 4h | `VERITAS_FANOUT=local` works; `=gemini` also works |
| 1.5 | **Prompt-injection hardening** + the poisoned-page test | 4h | Test proves a malicious page can't flip the verdict |
| 1.6 | Evidence panel in the UI: verbatim queries + snippets | 4h | I can read the exact query off the screen |
| 1.7 | Record eval fixtures now, before the swarm gets expensive | 3h | `evals/fixtures/` populated; CI replays them |

**Demo:** a compound claim splits into sub-claims; skeptic and steelman findings stream in
side by side with their real queries. **Total ~28h — heaviest sprint. Split 1.3/1.6 if it
overruns.**

---

## Sprint 2 — Judge, loop, pause · goal: *the money demo*

| # | Item | Box | Done when |
|---|------|-----|-----------|
| 2.1 | `quant` agent + `ContainerCodeExecutor`, hardened per ARCH §5 | 5h | Sandbox has no network; caps enforced; test proves it |
| 2.2 | `arbiter` in Chat mode: verdict + confidence per sub-claim | 4h | Output is schema-valid and cites findings |
| 2.3 | Confidence route + bounded `loop` node (max 2 iterations) | 3h | Loop is visible in the UI and terminates |
| 2.4 | HITL pause + Postgres session; `POST /claims/{id}/adjudicate` | 5h | Kill mid-pause, restart, adjudicate, graph resumes |
| 2.5 | `report` node + **citation validator** (dangling citation = hard fail) | 4h | Test proves an unresolvable citation raises |
| 2.6 | React Flow live graph: nodes light up as they execute | 5h | The graph animates during a real run |

**Demo:** the full ARCH §1 scenario through step 9. **Total ~26h.** 2.4 and 2.5 are the
items that make the interview claims true — do not let them slip.

---

## Sprint 3 — The eval · goal: *"here is the number"*

This sprint is the reason the repo exists. Treat it as un-cuttable.

| # | Item | Box | Done when |
|---|------|-----|-----------|
| 3.1 | Ingest an AVeriTeC/FEVER slice (100–150 claims); licensing checked | 4h | `make data` fetches or verifies hashes |
| 3.2 | Single-LLM baseline implementation (one call, one verdict) | 2h | Baseline scores on the same slice |
| 3.3 | Scoring harness: accuracy, **Brier score**, ECE, tokens, wall-clock | 5h | `make eval` prints the full table |
| 3.4 | `triage` cost router + its own eval (router precision per ARCH §6) | 5h | Table shows tokens saved and claims mis-routed |
| 3.5 | Calibration plot committed to `docs/` | 2h | Reliability diagram rendered |
| 3.6 | Eval floor asserted in CI — regressions fail the build | 3h | Lowering accuracy breaks CI |

**Demo:** `make eval` → the table. **Total ~21h.**

---

## Sprint 4 — Polish · goal: *"interview-ready"*

| # | Item | Box | Done when |
|---|------|-----|-----------|
| 4.1 | Dockerfile + `docker compose` (app + Postgres) | 3h | `docker compose up` runs the demo |
| 4.2 | README: eval table **first**, then architecture, honest capability list | 4h | A stranger gets it in 60 seconds |
| 4.3 | OTel traces; one screenshot of a trace in the README | 3h | Spans cover the whole graph |
| 4.4 | **"Explain it back to me"** — navigate and explain the repo cold, no LLM | 2h | I can trace a claim through all 13 nodes from memory |
| 4.5 | Record the 2-minute demo; finalise `DEMO_SCRIPT.md` | 3h | Video exists; script matches reality |

**Total ~15h.** Program total ≈ **108h**.

---

## Parking lot

Pull in only when a posting justifies it.

- A2A protocol: expose the arbiter as a remote agent another system can call
- MCP server for a private evidence corpus
- Multi-lingual claims
- Retrieval over previously-verified claims ("we checked this in March")
- Slack/Discord submission surface
