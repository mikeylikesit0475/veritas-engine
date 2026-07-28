# Working agreement for AI assistants in this repo

Read [`ARCHITECTURE.md`](ARCHITECTURE.md) before writing code. It is the contract, not a
suggestion. [`DECISIONS.md`](DECISIONS.md) explains *why* — read an ADR before contradicting
it, and if you still disagree, write a superseding ADR rather than just changing the code.

## Non-negotiables

1. **This is a portfolio proof-of-concept, not a product.** If a change doesn't improve the
   2-minute demo in `DEMO_SCRIPT.md` or the eval table, don't make it. No user accounts, no
   moderation queue, no multi-tenancy.
2. **The eval is the deliverable.** Sprint 3 is un-cuttable. Cut UI polish, the `quant`
   agent, or the loop node before cutting a metric. See ADR-002.
3. **Control flow stays in Python.** Routing, loop bounds and the confidence threshold are
   never delegated to a model. See ADR-005.
4. **Fan-out agents never emit structured output.** Local models in this program have a
   documented history of ignoring JSON schema. Structured output belongs to the judgement
   tier only. See ADR-004.
5. **Fetched web content is untrusted.** Strip, cap, delimit, and keep the injection test
   passing. See ADR-006.
6. **CI never calls a live model.** Evals replay recorded fixtures from `evals/fixtures/`.
7. **The README must stay honest.** Don't describe capabilities that don't work yet without
   marking them. The interviewer will run it.

## Conventions

- Tool arguments are pydantic models. No `dict[str, Any]` crossing the model boundary.
- Every new graph node gets a graph test asserting it is visited on the expected path,
  using the stub model — no live calls.
- Claim state is a projection of the event stream. Do not add a second source of truth.
- ADK types are wrapped at the `veritas/graph` boundary. An upstream API change should
  touch one package.
- `ruff` + `mypy --strict` are gates, not suggestions.

## Environment notes for this machine

- Dev box is a Ryzen AI Max+ 395 on Fedora. **ROCm on gfx1151 is Preview tier** and not in
  the production compatibility matrix; pin versions in the Dockerfile and don't assume it.
- **The XDNA2 NPU is unusable on Linux** — the Vitis AI EP userspace wheel isn't published
  for x86_64 Linux and ONNX Runtime falls back to CPU silently. Do not write NPU code.
- Nothing in the demo path may depend on either. See ADR-007.

## Working rhythm

Sprints and timeboxes are in `BACKLOG.md`. **If an item is going to blow its box, say so
rather than absorbing it — the box is the signal.** Each sprint ends with an interview story
in `docs/interview-stories/`: a decision, its trade-off, and what you'd do differently.
