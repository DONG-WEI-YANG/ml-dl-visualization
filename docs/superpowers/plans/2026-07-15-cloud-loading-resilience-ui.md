# Cloud Loading Resilience UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure students can enter local course content within eight seconds even when the cloud API is asleep or unavailable, while making backend health responsive during RAG initialization.

**Architecture:** A typed frontend request layer classifies network failures and an auth/service context separates identity verification from cloud availability. Offline-safe routes render immediately, while backend startup publishes health state and moves RAG ingestion to an asynchronous task.

**Tech Stack:** React 19, TypeScript, React Router 7, Vitest, Testing Library, FastAPI, pytest, asyncio, SQLite.

## Global Constraints

- Authentication verification timeout is exactly 8 seconds; the waking state begins after 3 seconds.
- Only an explicit HTTP 401 removes the stored token; timeout, offline, and 5xx failures preserve it.
- Home and week routes are offline-safe; dashboard and admin routes still require verified authentication.
- RAG initialization must not block `/health` from responding.
- No new runtime dependencies.

---

### Task 1: Typed frontend request failures

**Files:**
- Create: `platform/frontend/src/test/api.test.ts`
- Modify: `platform/frontend/src/lib/api.ts`

**Interfaces:**
- Produces: `APIErrorKind`, `APIError`, and `fetchAPI<T>(path, body?, token?, options?)` where options contains `timeoutMs?: number` and `signal?: AbortSignal`.

- [ ] **Step 1: Write failing tests**

Test that a 401 rejects with `kind === "unauthorized"`, a 503 with `kind === "server"`, and an aborted request with `kind === "timeout"`. Use `vi.stubGlobal("fetch", vi.fn(...))`, fake timers, and call `fetchAPI("/api/auth/me", undefined, "token", { timeoutMs: 10 })`.

- [ ] **Step 2: Verify red**

Run: `npm test -- src/test/api.test.ts`
Expected: FAIL because `APIError` and request options do not exist.

- [ ] **Step 3: Implement the request contract**

Add:

```ts
export type APIErrorKind = "unauthorized" | "timeout" | "offline" | "server" | "unknown";
export class APIError extends Error {
  constructor(public kind: APIErrorKind, message: string, public status?: number) {
    super(message);
    this.name = "APIError";
  }
}
export interface FetchAPIOptions { timeoutMs?: number; signal?: AbortSignal }
```

Create an internal `AbortController`, abort after `timeoutMs`, forward an external abort signal, classify 401/5xx/network errors, and always clear the timer and listener in `finally`.

- [ ] **Step 4: Verify green**

Run: `npm test -- src/test/api.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add platform/frontend/src/lib/api.ts platform/frontend/src/test/api.test.ts
git commit -m "feat: classify frontend API failures"
```

### Task 2: Resilient auth and loading experience

**Files:**
- Create: `platform/frontend/src/components/CloudLoadingState.tsx`
- Create: `platform/frontend/src/test/AuthResilience.test.tsx`
- Modify: `platform/frontend/src/hooks/useAuth.tsx`
- Modify: `platform/frontend/src/App.tsx`

**Interfaces:**
- Consumes: `APIError` and `fetchAPI` from Task 1.
- Produces: auth context fields `verification: "checking" | "authenticated" | "anonymous" | "unverified"`, `cloudStatus: "connecting" | "waking" | "ready" | "unavailable"`, and `retryVerification(): Promise<void>`.

- [ ] **Step 1: Write failing auth resilience tests**

With fake timers and a never-resolving fetch, seed `localStorage.auth_token`, render `App`, advance 3 seconds and assert `正在喚醒雲端服務`; advance to 8 seconds and assert `進入離線課程`. Add cases proving timeout/503 preserve the token and 401 removes it.

- [ ] **Step 2: Verify red**

Run: `npm test -- src/test/AuthResilience.test.tsx`
Expected: FAIL because the new states and Chinese actions are absent.

- [ ] **Step 3: Implement auth state separation**

Replace the single `loading` boolean with the explicit verification/cloud states. Start a 3-second waking timer, call `/api/auth/me` with `{ timeoutMs: 8000 }`, clear the token only for `APIError.kind === "unauthorized"`, and expose a retry function. Keep `loading` as a derived compatibility value if existing consumers require it.

- [ ] **Step 4: Implement `CloudLoadingState`**

Render the approved three-stage copy and actions. `先瀏覽課程` and `進入離線課程` must navigate to the offline-safe home route; `重新連線` calls `retryVerification`; `登出並返回登入` calls logout. Use semantic headings, buttons, `role="status"`, and `aria-live="polite"`.

- [ ] **Step 5: Update route protection**

Permit `/` and `/week/:weekId` while `verification === "unverified"`. Keep dashboard/admin pages behind verified authentication. Replace the plain full-screen `Loading...` with `CloudLoadingState`.

- [ ] **Step 6: Verify green and regression suite**

Run: `npm test -- src/test/AuthResilience.test.tsx src/test/App.test.tsx src/test/Home.test.tsx src/test/WeekPage.test.tsx`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add platform/frontend/src/components/CloudLoadingState.tsx platform/frontend/src/test/AuthResilience.test.tsx platform/frontend/src/hooks/useAuth.tsx platform/frontend/src/App.tsx
git commit -m "feat: allow offline-safe course access"
```

### Task 3: Degrade backend-dependent course panels locally

**Files:**
- Create: `platform/frontend/src/components/CloudFeatureGate.tsx`
- Create: `platform/frontend/src/test/CloudFeatureGate.test.tsx`
- Modify: `platform/frontend/src/pages/WeekPage.tsx`

**Interfaces:**
- Consumes: `cloudStatus` from auth context.
- Produces: `CloudFeatureGate` with `children`, `title`, and optional `onRetry` props.

- [ ] **Step 1: Write failing gate test**

Render the gate with `cloudStatus="unavailable"`; assert its children are absent and the text `等待雲端服務` plus retry button are present. Render with `ready`; assert children appear.

- [ ] **Step 2: Verify red**

Run: `npm test -- src/test/CloudFeatureGate.test.tsx`
Expected: FAIL because the component is absent.

- [ ] **Step 3: Implement the focused gate**

Render children only for `ready`; otherwise render a bordered status card explaining that the current feature requires cloud access. Keep course visualization content outside the gate. Wrap `ChatPanel`, `QuizPanel`, and curriculum download controls in `WeekPage`.

- [ ] **Step 4: Verify green**

Run: `npm test -- src/test/CloudFeatureGate.test.tsx src/test/WeekPage.test.tsx`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add platform/frontend/src/components/CloudFeatureGate.tsx platform/frontend/src/test/CloudFeatureGate.test.tsx platform/frontend/src/pages/WeekPage.tsx
git commit -m "feat: isolate cloud-only course features"
```

### Task 4: Non-blocking backend readiness

**Files:**
- Create: `platform/backend/app/readiness.py`
- Modify: `platform/backend/app/main.py`
- Modify: `platform/backend/tests/test_health.py`

**Interfaces:**
- Produces: module-level `readiness` object with `status`, `database`, `rag`, `started_at`, and `snapshot()`; coroutine `initialize_rag_background()`.

- [ ] **Step 1: Write failing health tests**

Update the baseline expectation to `status == "ready"`, `database == "connected"`, `rag` in the documented state set, and numeric `uptime_seconds`. Monkeypatch the RAG initializer with a waiting coroutine, start the lifespan, and assert `/health` responds without awaiting that coroutine. Add a failure case expecting `degraded` and `rag == "error"`.

- [ ] **Step 2: Verify red**

Run: `pytest tests/test_health.py -v`
Expected: FAIL because health has only `ok` and database fields.

- [ ] **Step 3: Implement readiness state**

Define a small dataclass protected by assignments on the event loop. `snapshot()` returns only the documented safe fields. In lifespan, complete `init_db()` and `init_rag_tables()`, mark the database connected, schedule RAG ingestion with `asyncio.create_task`, then mark the core service ready. Store the task and cancel/await it during shutdown.

- [ ] **Step 4: Move RAG ingestion behind the task boundary**

Convert `_auto_ingest_curriculum` into the scheduled worker. Set `rag="indexing"` before reading files, `ready` on success/empty data, and `error` plus overall `degraded` on exception. Keep daily web enrichment independently scheduled after core readiness.

- [ ] **Step 5: Replace health response**

Return `readiness.snapshot()` with status 200 when database is connected and 503 otherwise. The handler must not query RAG or external services.

- [ ] **Step 6: Verify green and backend regressions**

Run: `pytest tests/test_health.py -v`
Expected: PASS.

Run: `pytest -v`
Expected: all backend tests PASS.

- [ ] **Step 7: Commit**

```bash
git add platform/backend/app/readiness.py platform/backend/app/main.py platform/backend/tests/test_health.py
git commit -m "perf: make backend readiness non-blocking"
```

### Task 5: Full verification and deployment measurement

**Files:**
- Modify only if verification exposes a defect in files already listed above.

- [ ] **Step 1: Run frontend verification**

Run: `npm test`
Expected: all tests PASS.

Run: `npm run lint`
Expected: exit 0 with no errors.

Run: `npm run build`
Expected: TypeScript and Vite build succeed.

- [ ] **Step 2: Run backend verification**

Run: `pytest -v`
Expected: all tests PASS.

- [ ] **Step 3: Manually verify responsive states**

Run the frontend with a deliberately unreachable API base. At desktop and mobile widths, confirm the 3-second waking copy, 8-second offline entry, week navigation, disabled cloud panels, keyboard focus, and recovery after restoring the API.

- [ ] **Step 4: Re-measure cloud after deployment**

Run two timed requests to `/health`. Record DNS, TLS, first-byte, total time, and returned readiness fields. Confirm the second request is responsive and use the service logs to distinguish provider wake time from application initialization.

- [ ] **Step 5: Commit verification-only fixes if needed**

```bash
git add <only-files-changed-to-fix-verification>
git commit -m "fix: address resilience verification findings"
```
