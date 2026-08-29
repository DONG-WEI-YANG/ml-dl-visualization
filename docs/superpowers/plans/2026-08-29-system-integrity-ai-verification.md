# System Integrity and AI Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete ten evidence-driven loops that align identity, data, quiz, AI provider, streaming UI, and verification contracts.

**Architecture:** Retain FastAPI/SQLite and React hooks, adding validation at public boundaries and small focused diagnostics/lifecycle helpers. Each loop is an independently testable red-green-refactor cycle.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic 2, SQLite, pytest; React 19, TypeScript, Vitest, Testing Library, Vite.

**Spec:** `docs/superpowers/specs/2026-08-29-system-integrity-ai-verification-design.md`

## Global Constraints

- Keep the existing React/FastAPI architecture and visual language.
- Never expose API keys or raw provider exception text to the UI.
- Do not make paid external AI calls; real probing defaults to local and external providers require explicit configured credentials.
- Every behavior change follows RED → GREEN → REFACTOR and preserves the full suite.

---

### Task 1: Deterministic test and connection baseline

**Files:** Modify `platform/backend/pyproject.toml`, `platform/backend/app/db.py`, `platform/backend/tests/test_db.py`, `platform/frontend/package.json`, `platform/frontend/package-lock.json`, `platform/frontend/src/test/setup.ts`.

**Interfaces:** Produces `db_connection()` context manager and working backend/frontend coverage commands.

- [ ] Add a failing test that raises inside `with db_connection()` and asserts a subsequent operation proves the first connection was closed.
- [ ] Run `pytest tests/test_db.py -q` and confirm failure because `db_connection` does not exist.
- [ ] Implement `db_connection()` with commit-on-success, rollback-on-error, and unconditional close; migrate setting helpers to it.
- [ ] Add `asyncio_default_fixture_loop_scope = "function"`, install `@vitest/coverage-v8` matching Vitest, and centralize jsdom canvas/navigation-safe test shims.
- [ ] Run focused tests, `npm test`, and both coverage commands.

### Task 2: Learning-event data contract

**Files:** Modify `platform/backend/app/analytics/models.py`, `platform/backend/tests/test_analytics.py`.

**Interfaces:** `LearningEvent(student_id, week, event_type, topic, score, duration_seconds, metadata, timestamp)` rejects invalid domain values and uses per-instance collections.

- [ ] Add parameterized failing tests for week 0/19, unknown event type, score below 0/above 100, negative duration, and mutable-default isolation.
- [ ] Run the focused test and confirm Pydantic currently accepts invalid input.
- [ ] Add `Field`/`Literal` constraints and `default_factory` collections.
- [ ] Run focused and complete backend tests.

### Task 3: Identity-bound analytics and authorization

**Files:** Modify `platform/backend/app/api/analytics_routes.py`, `platform/backend/app/analytics/tracker.py`, create `platform/backend/tests/test_api_analytics_auth.py`.

**Interfaces:** `POST /api/analytics/events` derives student ID from authenticated user; analytics reads require self, assigned teacher, or admin.

- [ ] Add failing API tests for anonymous rejection, spoof prevention, self read, unrelated student denial, assigned teacher access, and admin summary access.
- [ ] Run tests and confirm the existing public endpoints/spoof behavior fail the contract.
- [ ] Add authentication dependencies and a focused authorization helper; keep database reads closed through all branches.
- [ ] Run focused and full backend tests.

### Task 4: Complete and consistent quiz grading

**Files:** Modify `platform/backend/app/quiz/questions.py`, `platform/backend/app/api/quiz_routes.py`, `platform/backend/tests/test_api_quiz.py`, `platform/frontend/src/components/quiz/QuizPanel.tsx`, `platform/frontend/src/test/QuizPanel.test.tsx`.

**Interfaces:** Grade results always contain every stored question with `user_answer`, `correct_answer`, `correct`, `explanation`; total equals stored question count.

- [ ] Add failing tests for partial submission, unknown IDs, negative/out-of-range option indexes, empty week, and response field names.
- [ ] Confirm failures show the current submitted-answer denominator and schema mismatch.
- [ ] Implement database-authoritative grading and Pydantic answer validation; add visible frontend load/submit errors.
- [ ] Run backend and frontend focused tests, then both full suites.

### Task 5: Provider resolution and truthful diagnostics

**Files:** Modify `platform/backend/app/llm/factory.py`, `platform/backend/app/api/llm_routes.py`, create `platform/backend/tests/test_llm_diagnostics.py`.

**Interfaces:** `resolve_llm_provider()` returns provider plus configured/effective metadata; `GET /api/llm/diagnostics` returns safe status and optional probe result.

- [ ] Add failing tests for local, missing OpenAI key fallback, unknown provider fallback, and authenticated admin diagnostics without secret leakage.
- [ ] Run focused tests and confirm no resolution metadata/endpoint exists.
- [ ] Implement a typed resolution record and diagnostics endpoint; make model-info use the same source of truth.
- [ ] Execute a local provider smoke probe against the real pipeline and run the focused/full tests.

### Task 6: Real personalization across HTTP and WebSocket

**Files:** Modify `platform/backend/app/llm/tutor.py`, `platform/backend/app/api/llm_routes.py`, `platform/backend/tests/test_llm.py`, `platform/backend/tests/test_llm_streaming.py`.

**Interfaces:** `AITutor.ask/ask_stream(..., student_id=)` forwards identity into `_build_system`; chat routes use authenticated payload id.

- [ ] Add failing tests that record an event and assert the provider receives a system prompt containing the student learning context for both transports.
- [ ] Confirm failures because routes omit `student_id`.
- [ ] Thread the id through HTTP and WebSocket orchestration without accepting it from client payload.
- [ ] Run focused and full backend tests.

### Task 7: Provider response and streaming hardening

**Files:** Modify `platform/backend/app/llm/openai_provider.py`, `anthropic_provider.py`, `ollama_provider.py`, create `platform/backend/tests/test_llm_providers.py`.

**Interfaces:** Providers return non-empty text or raise a sanitized provider error; Ollama calls `raise_for_status()` and ignores malformed/terminal empty frames safely.

- [ ] Add failing contract tests using complete provider-shaped fakes for empty content, usage absence, HTTP 500, malformed NDJSON, and valid chunks.
- [ ] Confirm each failure maps to a missing guard in current adapters.
- [ ] Add minimal normalization and status checks without changing provider APIs.
- [ ] Run provider and complete backend tests.

### Task 8: Resilient chat lifecycle and visible errors

**Files:** Modify `platform/frontend/src/hooks/useChat.ts`, `platform/frontend/src/components/llm/ChatPanel.tsx`, `platform/frontend/src/test/ChatPanel.test.tsx`.

**Interfaces:** `useChat` exposes `error`, `retry`, `stop`; at most one socket is active; malformed messages and premature close result in a visible recoverable state.

- [ ] Add failing behavior tests for server error content, invalid JSON, premature close after draft, stop generation, resend closing the prior socket, and retrying the last question.
- [ ] Confirm failures because the hook discards errors and has no stop/retry contract.
- [ ] Implement lifecycle cleanup and render a compact aria-live error row with Retry/Stop controls.
- [ ] Run focused and complete frontend tests.

### Task 9: AI diagnostics UI and synchronized admin settings

**Files:** Modify `platform/frontend/src/pages/AdminSettings.tsx`, create `platform/frontend/src/pages/__tests__/AdminSettings.test.tsx`, modify `platform/frontend/src/lib/api.ts`.

**Interfaces:** Admin UI displays configured → effective provider, status/reason, last probe result; failed setting saves do not leave optimistic stale state.

- [ ] Add failing tests for ready local, degraded fallback, probe failure, settings save error, and accessible status labels.
- [ ] Confirm the current page has no truthful runtime status.
- [ ] Add typed diagnostics fetching/probing and a restrained status card following the existing visual system.
- [ ] Run focused/full tests, lint, and build.

### Task 10: Coverage convergence, lint synchronization, and bundle integration

**Files:** Modify only files identified by fresh coverage/lint evidence, including Hook callback boundaries and `platform/frontend/src/App.tsx` lazy imports; update `README.md` and this plan checklist.

**Interfaces:** Route components load through `React.lazy`/`Suspense`; lint has zero warnings; verification commands are documented.

- [ ] Run fresh backend/frontend coverage and name behavior-relevant uncovered branches before adding tests.
- [ ] Add failing tests for chosen uncovered error/boundary branches and verify each RED.
- [ ] Fix Hook dependency warnings with stable callbacks, isolate non-component exports where required, and lazy-load heavy routes/components.
- [ ] Run complete backend/frontend suites, coverage, lint, build, local AI smoke, data-integrity query, and `git diff --check`.
- [ ] Record each loop's result and remaining justified limitations in the final report; do not claim external paid providers were called without credentials.
