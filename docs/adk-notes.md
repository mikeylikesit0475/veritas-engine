# ADK spike notes

Spike findings from reading the installed ADK source. Each claim is verified against
`google-adk==2.5.0`, not assumed. When a future ADK version lands, re-run these checks.

---

## Sprint 0.3 — Postgres sessions + HITL resume

**Question.** Does ADK Python ship a Postgres-backed session service with HITL resume?
This is load-bearing for Sprint 2.4 ("kill the process mid-pause, restart, adjudicate,
graph resumes") — the strongest claim in the demo.

**Answer.** Yes. Both pieces ship in `google-adk==2.5.0`.

### Verified components

| Claim | Verified at | Notes |
|-------|-------------|-------|
| `DatabaseSessionService` exists | `google.adk.sessions.DatabaseSessionService` | Exported from the public `sessions` package. |
| Postgres URL accepted | `DatabaseSessionService.__init__(db_url=...)` | URL passes through SQLAlchemy `make_url`; use `postgresql+asyncpg://...` for the async driver. Mutually exclusive with `db_engine`. |
| Async engine is the only mode | ctor signature + `AsyncEngine` typing | There is no sync variant. The pasted answer's "ensure async driver" note is correct — a sync `postgresql://` URL would mismatch. |
| Events persisted, not just session state | `google.adk.sessions.schemas.v1` tables: `sessions`, `events`, `app_states`, `user_states` | The `events` table is the load-bearing one for HITL: a pending `adk_request_confirmation` function call is an event row, so it survives a restart. |
| HITL is a synthetic function call | `REQUEST_CONFIRMATION_FUNCTION_CALL_NAME == "adk_request_confirmation"` in `google.adk.flows.llm_flows.request_confirmation` | The pasted answer's name was exact. There is also a richer `request_input` path (`REQUEST_INPUT_FUNCTION_CALL_NAME`) for structured human input. |
| Resume via `session_id` + `FunctionResponse` | `google.adk.workflow.utils._workflow_hitl_utils`: `has_request_input_function_call`, `get_request_input_interrupt_ids`, `process_auth_resume` | Resume = call `runner.run_async` again with the same `session_id` and a `FunctionResponse` for the pending interrupt id. The runner pulls serialized context from Postgres and continues. |

### The resume contract, concretely

1. Graph hits a `ToolConfirmation` boundary (or `request_input`). The runner emits an
   `Event` containing a function call to `adk_request_confirmation` (or
   `adk_request_input`) with an `interrupt_id`.
2. `DatabaseSessionService` has already committed that event row to Postgres. The Python
   process can now exit — no blocking thread is held.
3. On restart, the service rehydrates the session from `sessions` + `events`.
4. The application issues a `FunctionResponse` tied to the `interrupt_id` and calls
   `runner.run_async(session_id=<existing>, new_message=<that response>)`. The graph
   continues from the paused node.

### Implications for Veritas (Sprint 2.4)

- **The kill/restart demo is buildable as specified.** No custom session persistence code
  is needed — `DatabaseSessionService` is the whole seam. We wrap it at the
  `veritas/server` boundary per ADR-001.
- **Postgres is required for the demo.** ADR-008 (Postgres, not SQLite) is now
  *enforced* by the demo's central claim, not just a portfolio choice. The `SessionService`
  interface is still the seam, but the demo specifically depends on Postgres's
  durability — SQLite's single-writer model would also work technically but the
  interview claim is stronger with Postgres.
- **The HITL node (ARCH §3, node 12) is a `ToolConfirmation` or `request_input` call**,
  not a hand-rolled pause. This is good — it means the pause/resume is ADK-native and
  we don't fight the framework.
- **One thing to size in Sprint 2.4:** the `POST /claims/{id}/adjudicate` endpoint must
  look up the pending `interrupt_id` from the session's event history and construct the
  `FunctionResponse`. ~1h of the 5h box. The rest is wiring + the kill/restart test.

### Risk that did *not* materialise

- "ADK's HITL only works in the built-in `adk web` server, not a custom FastAPI app."
  **False.** `DatabaseSessionService` and the runner are framework-agnostic; the FastAPI
  app in `veritas/server` calls the runner directly. No `adk web` dependency.

### Risk that *did* materialise (minor)

- `DatabaseSessionService` requires the `db` extra (`sqlalchemy` + `asyncpg`). Already
  satisfied: `google-adk[extensions]` pulled `sqlalchemy` transitively; we need to add
  `asyncpg` explicitly when we wire the Postgres URL in Sprint 0.5/2.4.

**Spike conclusion: green. Proceed with Sprint 2.4 as planned.**