# AI Tutor First Response Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a useful local AI tutor draft within one second, then replace it with the existing fully verified NLP/RAG response.

**Architecture:** A focused `quick_answer` module produces a bounded course-only draft without importing the full NLP pipeline. The WebSocket route orchestrates status, draft, refinement, and done events; the React chat hook maps those events into one progressively refined assistant message.

**Tech Stack:** FastAPI WebSocket, asyncio, Python, React 19, TypeScript, Vitest, pytest.

## Global Constraints

- Draft timeout is 800ms and local draft p95 target is below 1,000ms.
- Draft uses at most two current-week RAG chunks and 1,200 context characters.
- Existing full RAG and 42-layer NLP pipeline remains the final answer path.
- Existing `chunk` events remain frontend-compatible.
- No new runtime dependencies and no persistent WebSocket prewarming.

---

### Task 1: Bounded quick-answer unit

**Files:**
- Create: `platform/backend/app/llm/quick_answer.py`
- Create: `platform/backend/tests/test_quick_answer.py`

**Interfaces:**
- Produces: `build_quick_answer(question: str, week: int, topic: str) -> str | None`.

- [ ] Write failing tests proving an in-course question returns 2–4 sentences, the retriever is called with `top_k=2`, output is bounded, and high-risk/out-of-scope prompts return a conservative scope message rather than a factual guess.
- [ ] Run `python -m pytest tests/test_quick_answer.py -v`; expect failure because the module is absent.
- [ ] Implement keyword-based intent framing plus `retrieve_context(question, week, top_k=2, max_context_chars=1200)`. Do not import `app.nlp`.
- [ ] Run the focused tests; expect all pass.
- [ ] Commit with `feat: add bounded AI quick answers`.

### Task 2: Two-stage WebSocket orchestration

**Files:**
- Modify: `platform/backend/app/api/llm_routes.py`
- Create: `platform/backend/tests/test_llm_streaming.py`
- Modify: `platform/backend/app/rag/retriever.py`

**Interfaces:**
- Consumes: `build_quick_answer` from Task 1.
- Produces events `status/analyzing`, optional `draft`, `status/verifying`, `refinement`, `done`, and staged `error`.

- [ ] Write a WebSocket test with monkeypatched quick answer and tutor stream, asserting exact event order and timing fields; add draft failure and 800ms timeout cases that still reach refinement.
- [ ] Run `python -m pytest tests/test_llm_streaming.py -v`; expect failure because existing events are only `chunk/done`.
- [ ] Add optional `max_context_chars` to `retrieve_context` while keeping its current 4,000 default.
- [ ] In `chat_ws`, send analyzing immediately, run quick answer through `asyncio.to_thread` under `asyncio.wait_for(..., 0.8)`, send draft when present, send verifying, relay provider chunks as refinement, and log only provider/week/timings.
- [ ] Run focused and complete backend tests; expect pass.
- [ ] Commit with `feat: stream AI drafts before verified answers`.

### Task 3: Progressive chat message UI

**Files:**
- Modify: `platform/frontend/src/hooks/useChat.ts`
- Modify: `platform/frontend/src/components/llm/ChatPanel.tsx`
- Modify: `platform/frontend/src/test/ChatPanel.test.tsx`

**Interfaces:**
- Consumes the Task 2 event contract.
- Produces chat state `stage: "idle" | "analyzing" | "draft" | "verifying" | "verified" | "unverified"`.

- [ ] Add failing hook/panel tests that simulate status, draft, refinement, done, refinement error, and legacy chunk events. Assert the first refinement replaces the draft in the same assistant bubble.
- [ ] Run `npm test -- src/test/ChatPanel.test.tsx`; expect failures for missing stages and labels.
- [ ] Track draft and refinement buffers separately in `useChat`; create only one assistant message, replace on first refinement, preserve draft on refinement failure, and expose `stage`.
- [ ] Render immediate Chinese stage labels and verified/unverified badges in `ChatPanel` without duplicating the loading bubble.
- [ ] Run focused tests, full frontend tests, lint, and build; expect zero errors.
- [ ] Commit with `feat: show progressive AI answer verification`.

### Task 4: Final verification

**Files:**
- Modify only files above if verification exposes a defect.

- [ ] Run `npm test`, `npm run lint`, and `npm run build` in `platform/frontend`.
- [ ] Run `python -m pytest -q` in `platform/backend`.
- [ ] Benchmark 100 local calls to `build_quick_answer`; assert p95 is below 1,000ms and record the result.
- [ ] Review `git diff --check`, status, and commits before integration.
