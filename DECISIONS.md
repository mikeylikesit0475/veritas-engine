# Architecture Decision Records — Veritas Engine

Format: context → decision → consequence. An ADR is only superseded by another ADR.

---

## ADR-001 — Python + ADK, not a hand-rolled agent loop

**Context.** The graph could be written with plain async Python and direct model calls.
That would be less code.

**Decision.** Use `google-adk` graph workflows.

**Consequence.** We inherit HITL suspension, session persistence and event streaming rather
than building them. We also inherit a young API surface — mitigated by pinning the version
and wrapping ADK types at the `veritas/graph` boundary so an upstream change touches one
package. The portfolio reason is decisive: this repo's job is to demonstrate ADK fluency.

---

## ADR-002 — The eval is the product; the architecture is the packaging

**Context.** Multi-agent debate demos are abundant. Ours is not architecturally novel.

**Decision.** Sprint 3 (the eval) is un-cuttable. The README leads with the eval table, not
the architecture diagram. If schedule pressure forces a cut, we cut the UI polish, the
`quant` agent, and the loop node before we cut a single eval metric.

**Consequence.** The repo makes a falsifiable claim. That is rarer than a clever graph and
it is the thing an interviewer can push on productively.

---

## ADR-003 — Skeptic and steelman are separate agents, not one balanced agent

**Context.** A single agent instructed to "weigh both sides" is cheaper.

**Decision.** Two agents with opposed, one-sided instructions; a third node reconciles.

**Consequence.** More tokens. In exchange, each search is actually adversarial — a single
balanced agent tends to find one supporting source, one contradicting source, and declare
uncertainty regardless of the evidence's real weight. The eval in Sprint 3 must confirm
this is worth its cost; **if it isn't, this ADR gets superseded and we say so in the
README.** That negative result would itself be a good interview story.

---

## ADR-004 — Fan-out agents never emit structured output

**Context.** Local and Ollama-hosted models have repeatedly failed to honour JSON schema in
this program (see the Vane work and `incident-copilot` ADR on the same subject). We want
the high-volume fan-out to run on a local model for cost reasons.

**Decision.** Structured output is confined to the judgement tier (`triage`, `decompose`,
`arbiter`), which runs on Gemini. Fan-out agents return prose; a deterministic parser or a
judgement-tier call structures it.

**Consequence.** The local tier can be swapped, degraded or removed without breaking the
graph's control flow. Schema failures surface as loud validation errors, never as silently
wrong routes. Costs one extra structuring call per fan-out batch.

---

## ADR-005 — Control flow stays in Python

**Context.** Routing, loop bounds and confidence thresholds could all be delegated to the
arbiter's judgement.

**Decision.** They are not. `route`, `loop` and the iteration bound are ordinary Python.
The model supplies a confidence number; Python decides what to do with it.

**Consequence.** The path a claim took is unit-testable and reproducible. "I can assert
which nodes ran, in order, without calling a model" is the strongest thing this repo can
say to an engineer reading it.

---

## ADR-006 — Fetched web content is untrusted input

**Context.** The system reads arbitrary pages found by search. Those pages can contain
instructions aimed at the model.

**Decision.** Fetched content is script-stripped, size-capped, and wrapped in a delimited
block with an explicit data-not-instruction preamble. A test feeds a page containing an
override attempt and asserts the verdict is unchanged.

**Consequence.** Real defence is partial — prompt injection is not solved and the README
will say so. What the repo demonstrates is that the threat was modelled, mitigated at the
architectural boundary rather than in instruction text, and covered by a regression test.

---

## ADR-007 — Local model tier is optional, never load-bearing

**Context.** The dev box (Ryzen AI Max+ 395, ~110 GB addressable unified memory) can host a
large local model. But ROCm on gfx1151 is Preview tier, absent from the production support
matrix, with an open GPU-hang issue under mixed load. The XDNA2 NPU is unusable on Linux —
the Vitis AI EP userspace wheel is not published for x86_64 Linux, so ONNX Runtime falls
back to CPU silently.

**Decision.** `VERITAS_FANOUT=gemini` is the default and must always work end to end.
`=local` is an opt-in cost optimisation. No demo step depends on the local tier or the NPU.

**Consequence.** A driver regression on the dev box cannot break the interview demo. The
local path still earns its place in the README as a measured cost story.

---

## ADR-008 — Postgres sessions, not SQLite

**Context.** `incident-copilot` uses SQLite for the same purpose and it is sufficient there.

**Decision.** Postgres here.

**Consequence.** Slightly heavier `docker compose`. In exchange the two repos demonstrate
different persistence choices rather than looking copy-pasted, and Postgres is the answer
an enterprise interviewer expects for a multi-writer session store. The `SessionService`
interface is the seam, so this is reversible in an afternoon.

---

## ADR-009 — Dual model backend: Ollama default, Gemini optional

**Context.** ADR-004 confines structured output to a Gemini judgement tier, and ADR-007
makes `VERITAS_FANOUT=gemini` the must-work default. On the dev box, however, there is no
Gemini API key; the operator runs a substantial Ollama fleet (local MoEs at 50–75 GB plus
`:cloud` frontier models routed through Ollama: `glm-5.1:cloud`, `kimi-k2.7-code:cloud`,
`nemotron-3-super:cloud`). Forcing a Gemini key as a hard prerequisite would make the demo
non-functional on this machine and contradict the "must always work end to end" spirit of
ADR-007 — the *opposite* of what that ADR intended.

Two concerns from ADR-004 remain real and must be addressed, not waved away:

1. **Structured-output reliability.** The original concern was *local small* models
   ignoring JSON schema. The `:cloud` Ollama models are frontier-class, but they are not
   the Gemini API and we have no a priori proof they honour ADK's structured-output
   contract. ADR-004's mitigation — "schema validation fails loudly, never silently" —
   still applies and is now *more* important, not less.
2. **The interview story.** "We measured that local models blow schema, so we confined
   structured output to Gemini" was a clean claim. Dropping Gemini weakens it. The
   replacement claim must be honest and falsifiable: "we run local-first; here is what
   broke and what we did about it."

**Decision.**

- `veritas/models.py` is a factory with two backends, selected by environment:
  - `VERITAS_JUDGEMENT` — `ollama` (default) | `gemini`
  - `VERITAS_FANOUT` — `ollama` (default) | `gemini`
- **Ollama is the default on this machine.** No Gemini key is required to run the demo.
- **Gemini remains a supported, tested path** — `VERITAS_JUDGEMENT=gemini
  VERITAS_FANOUT=gemini` must work end to end. The factory never special-cases away the
  Gemini tier; it only changes the default.
- ADK's `LiteLlm(model="ollama_chat/<model>")` is the Ollama seam. The judgement tier on
  Ollama uses a `:cloud` model (e.g. `glm-5.1:cloud` or `kimi-k2.7-code:cloud`); the
  fan-out tier uses a local model. Model names are config, not code.
- **ADR-004's invariant is preserved and strengthened:** fan-out agents still never emit
  structured output. The judgement tier still does, and its schema validation must fail
  loudly on *either* backend. A regression test (Sprint 1.5 / eval) must exercise a
  judgement-tier structured-output call on the configured backend and assert the schema
  holds — or fail the build. If a chosen Ollama model proves unreliable on schema, that
  is a measured finding recorded in the README, not a silent degradation.
- **ADR-007 is amended, not superseded:** the "must always work end to end" default is
  Ollama on this box, Gemini where a key is present. The non-load-bearing principle
  stands — neither the local hardware nor the NPU may be a demo prerequisite.

**Consequence.**

- The demo runs on this machine today, with no external key. That is the point.
- We lose the clean "Gemini for schema" story and gain a "local-first, measured trade-offs"
  story. The eval in Sprint 3 must now include a **structured-output reliability row** in
  the table (schema-valid rate per backend) so the trade-off is falsifiable. If the local
  judgement model blows schema at a rate that hurts accuracy, that is a finding worth the
  README — and possibly grounds to supersede this ADR with a "Gemini judgement tier
  restored" ADR, with the measurement as evidence.
- More moving parts in `veritas/models.py`: two backends, env-driven selection, a stub
  fallback for CI. Wrapped at the `veritas/graph` boundary per ADR-001 so an ADK or
  LiteLLM upstream change still touches one package.
- CI invariant unchanged: no live model calls, ever. The stub-model harness (Sprint 0.6)
  is the only model CI ever sees.
