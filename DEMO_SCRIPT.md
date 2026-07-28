# Veritas Engine — 2-Minute Demo Script

Performed live, no code edits, no retries. If a step can't be performed, it is removed
from this file until it can — **this document must never describe a capability that
doesn't work.**

## Pre-flight (before the call)

```bash
docker compose up -d          # app + postgres
make seed-eval                # eval results cached so step 6 is instant
open http://localhost:3000
```

Check: airplane-mode equivalent — confirm `VERITAS_FANOUT` is set to whichever tier you
intend to talk about. Have `docs/reliability-diagram.png` open in a second tab.

---

## The script

**[0:00] Frame it in one sentence.**

> "This is an adversarial fact-checker built on Google's Agent Development Kit. Multi-agent
> debate demos are common, so the part I actually want to show you is the eval at the end."

**[0:10] Paste the claim.**

Use the rehearsed claim from `evals/demo_claim.txt` — one that reliably routes adversarial
and has genuinely mixed evidence.

**[0:15] Triage.**

Point at the router's output on screen.

> "First node is a cost router. It decided this one is worth the expensive path, and it
> says why. Easy claims take a single grounded call — I'll show the token difference later."

**[0:25] Decomposition.**

Three to five sub-claim nodes appear in the graph.

> "The compound claim is now atomic sub-claims that can each be checked independently."

**[0:35] The fan-out — let it run visibly.**

> "Each sub-claim gets a skeptic that only looks for refutation, a steelman that only looks
> for support, and where there's a number involved, an agent that writes Python and runs it
> in a sandbox with no network access."

**Scroll the evidence panel.** This is the credibility beat.

> "Those are the actual search queries it issued, and that's the actual Python source and
> its stdout. Nothing here is a summary of a summary."

**[0:55] Arbitration and the loop.**

> "The arbiter weighs all of it. Confidence came out under threshold on this sub-claim, so
> the graph looped once with new search directives — bounded at two iterations, in Python,
> not left to the model's judgement."

**[1:10] The kill test.** *(The step that proves the persistence claim.)*

Graph pauses at HITL. In a terminal:

```bash
docker compose restart app
```

Reload the browser. The pause is still there.

> "The graph suspended and the session is in Postgres. I just restarted the service
> mid-pause and it resumed where it was. That's the difference between a script and a
> workflow engine."

Click **Adjudicate**. The graph continues to the report.

**[1:25] The report.**

> "Every statement resolves to a stored citation. If one didn't, the render would fail —
> there's a test for that."

**[1:35] The eval. Slow down here.**

```bash
make eval
```

> "Held-out slice of AVeriTeC. Top row is a single Gemini call. Bottom row is this system.
> It's **N points** more accurate and costs **6×** the tokens."

Switch to the reliability diagram.

> "And this is the part most demos skip — calibration. When it says 80% confident, it's
> right about 80% of the time. A confident wrong answer is worse than an uncertain one, so
> I measured it."

**[1:55] Close.**

> "The router exists because of that cost multiple. It's measured too — of the claims it
> sent down the cheap path, none had their verdict changed by the expensive one."

---

## Questions to have answers ready for

| Question | Where the answer lives |
|----------|------------------------|
| "What stops a malicious page from steering the verdict?" | ADR-006 + the injection test. Answer honestly: mitigated, not solved. |
| "Why not one balanced agent instead of two?" | ADR-003 — and the eval number that justifies or refutes it. |
| "How do you know the confidence isn't made up?" | The reliability diagram. That's exactly what it measures. |
| "What would you do differently?" | `docs/interview-stories/`. Have one real answer, not a humble-brag. |
| "Does it work offline?" | Fan-out tier can run locally; grounding can't. Say so plainly. |
