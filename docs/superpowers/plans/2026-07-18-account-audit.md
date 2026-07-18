# Account Management Hardening + Audit Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add audit logging (admin actions + login history), self-service/forced password change, CSV batch user import, soft delete, semester archiving, and an admin audit UI; fix two High-severity auth gaps (unauthenticated LLM endpoints, default admin password).

**Architecture:** New `audit_logs` SQLite table + `log_audit()` helper called explicitly from routes. New `app/api/audit_routes.py` for query/export. Frontend gets a new AuditLog admin page, a ChangePasswordDialog (normal + forced mode), and UserManagement additions. All queries exclude soft-deleted users.

**Tech Stack:** FastAPI + sqlite3 (stdlib), pytest + TestClient, React 19 + TypeScript, vitest + @testing-library/react.

**Spec:** `docs/superpowers/specs/2026-07-18-account-audit-design.md`

## Global Constraints

- Audit write failures must never break the main operation: catch, `logging.warning`, continue (no silent `pass`).
- `detail` JSON must NEVER contain passwords or password hashes — only field names / counts.
- New password minimum length: 8 characters. Batch-import initial passwords: 12 chars from `secrets.token_urlsafe`.
- Soft delete only: `deleted_at` set, `is_active=0`; **never** delete rows from `learning_events` or `users`.
- All user-facing strings in Traditional Chinese (繁體中文), matching existing style.
- Audit page size cap: 200. CSV export: UTF-8 with BOM.
- Action names exactly as in the spec's Action Catalog (e.g. `login.success`, `user.delete`, `semester.archive`).
- Frontend uses `fetchAPI` from `lib/api.ts` where the HTTP method fits (GET no-body / POST with body); follow existing component style (Tailwind classes, no new UI libraries).
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

## Task 1: audit_logs table, users migration, log_audit helper

**Files:**
- Modify: `platform/backend/app/db.py` (init_db)
- Create: `platform/backend/app/audit/__init__.py`
- Test: `platform/backend/tests/test_audit.py`

**Interfaces:**
- Produces: `log_audit(action: str, *, actor: dict | None = None, target_type: str = "", target_id: str | int = "", detail: dict | None = None, ip: str = "") -> None`
- Produces: DB columns `users.deleted_at TEXT`, `users.must_change_password INTEGER NOT NULL DEFAULT 0`; table `audit_logs` with indexes.

- [ ] **Step 1: Write the failing test**

```python
# platform/backend/tests/test_audit.py
"""Tests for the audit logging helper and schema."""
from app.audit import log_audit
from app.db import get_db


def _fetch_all_audit():
    conn = get_db()
    rows = conn.execute("SELECT * FROM audit_logs ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def test_audit_table_exists_with_indexes():
    conn = get_db()
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    indexes = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
    conn.close()
    assert "audit_logs" in tables
    assert {"idx_audit_timestamp", "idx_audit_actor", "idx_audit_action"} <= indexes


def test_users_has_new_columns():
    conn = get_db()
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
    conn.close()
    assert "deleted_at" in cols
    assert "must_change_password" in cols


def test_log_audit_writes_row():
    actor = {"id": 1, "username": "admin", "role": "admin"}
    log_audit("user.update", actor=actor, target_type="user", target_id=42,
              detail={"fields": ["email"]}, ip="1.2.3.4")
    rows = [r for r in _fetch_all_audit() if r["action"] == "user.update"]
    assert rows, "audit row not written"
    row = rows[-1]
    assert row["actor_id"] == 1
    assert row["actor_username"] == "admin"
    assert row["target_type"] == "user"
    assert row["target_id"] == "42"
    assert '"email"' in row["detail"]
    assert row["ip"] == "1.2.3.4"


def test_log_audit_without_actor():
    log_audit("login.failed", detail={"username": "ghost"}, ip="5.6.7.8")
    row = [r for r in _fetch_all_audit() if r["action"] == "login.failed"][-1]
    assert row["actor_id"] is None
    assert row["actor_username"] == ""


def test_log_audit_never_raises(monkeypatch):
    import app.audit as audit_mod

    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(audit_mod, "get_db", boom)
    # Must not raise
    log_audit("user.create", detail={})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/backend && python -m pytest tests/test_audit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.audit'`

- [ ] **Step 3: Add schema to db.py**

In `platform/backend/app/db.py`, append inside the `conn.executescript("""...""")` block (after `quiz_questions`):

```sql
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            actor_id INTEGER,
            actor_username TEXT NOT NULL DEFAULT '',
            actor_role TEXT NOT NULL DEFAULT '',
            action TEXT NOT NULL,
            target_type TEXT NOT NULL DEFAULT '',
            target_id TEXT NOT NULL DEFAULT '',
            detail TEXT NOT NULL DEFAULT '{}',
            ip TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_logs(actor_id);
        CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);
```

Then, right after the existing semester try/except migration, add two more (same pattern):

```python
    # Migration: soft-delete + forced password change columns
    for ddl in (
        "ALTER TABLE users ADD COLUMN deleted_at TEXT",
        "ALTER TABLE users ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0",
    ):
        try:
            conn.execute(ddl)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
```

- [ ] **Step 4: Create the helper**

```python
# platform/backend/app/audit/__init__.py
"""Audit logging helper — explicit calls from routes; failures never break the caller."""
import json
import logging

from app.db import get_db

logger = logging.getLogger(__name__)


def log_audit(
    action: str,
    *,
    actor: dict | None = None,
    target_type: str = "",
    target_id: str | int = "",
    detail: dict | None = None,
    ip: str = "",
) -> None:
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO audit_logs (actor_id, actor_username, actor_role, action, "
            "target_type, target_id, detail, ip) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                actor["id"] if actor else None,
                actor["username"] if actor else "",
                actor["role"] if actor else "",
                action,
                target_type,
                str(target_id),
                json.dumps(detail or {}, ensure_ascii=False),
                ip,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:  # noqa: BLE001 — audit must never break the main operation
        logger.warning("audit log write failed for %s: %s", action, e)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd platform/backend && python -m pytest tests/test_audit.py -v`
Expected: PASS (5 tests). Note: the temp test DB is created once per session by conftest; `init_db()` runs the new migrations there.

- [ ] **Step 6: Commit**

```bash
git add platform/backend/app/db.py platform/backend/app/audit/__init__.py platform/backend/tests/test_audit.py
git commit -m "feat: add audit_logs schema, user soft-delete columns, log_audit helper"
```

---

## Task 2: Audit query + CSV export API

**Files:**
- Create: `platform/backend/app/api/audit_routes.py`
- Modify: `platform/backend/app/main.py` (register router — find the existing `app.include_router(...)` block and add one line)
- Test: `platform/backend/tests/test_api_audit.py`

**Interfaces:**
- Consumes: `log_audit` (Task 1), `require_admin` from `app.auth.dependencies`.
- Produces: `GET /api/admin/audit-logs` → `{"items": [...], "total": int, "page": int, "page_size": int}`; `GET /api/admin/audit-logs/export` → CSV.

- [ ] **Step 1: Write the failing test**

```python
# platform/backend/tests/test_api_audit.py
"""Integration tests for audit log query/export endpoints."""
from fastapi.testclient import TestClient
from app.main import app
from app.audit import log_audit

client = TestClient(app)


def _admin_token() -> str:
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _seed():
    actor = {"id": 1, "username": "admin", "role": "admin"}
    log_audit("user.create", actor=actor, target_type="user", target_id=7, ip="9.9.9.9")
    log_audit("settings.update", actor=actor, target_type="setting", detail={"keys": ["llm_provider"]})
    log_audit("login.failed", detail={"username": "ghost"}, ip="8.8.8.8")


def test_list_requires_admin():
    resp = client.get("/api/admin/audit-logs")
    assert resp.status_code in (401, 403)


def test_list_returns_paginated_items():
    _seed()
    token = _admin_token()
    resp = client.get("/api/admin/audit-logs?page=1&page_size=2", headers=_auth(token))
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 3
    assert len(data["items"]) == 2
    # newest first
    assert data["items"][0]["id"] > data["items"][1]["id"]


def test_filter_by_action_prefix():
    _seed()
    token = _admin_token()
    resp = client.get("/api/admin/audit-logs?action_prefix=login", headers=_auth(token))
    assert resp.status_code == 200
    assert all(item["action"].startswith("login") for item in resp.json()["items"])
    assert resp.json()["total"] >= 1


def test_filter_by_actor():
    _seed()
    token = _admin_token()
    resp = client.get("/api/admin/audit-logs?actor_id=1", headers=_auth(token))
    assert all(item["actor_id"] == 1 for item in resp.json()["items"])


def test_page_size_capped_at_200():
    token = _admin_token()
    resp = client.get("/api/admin/audit-logs?page_size=999", headers=_auth(token))
    assert resp.status_code == 422  # ge/le validation


def test_export_csv():
    _seed()
    token = _admin_token()
    resp = client.get("/api/admin/audit-logs/export?action_prefix=login", headers=_auth(token))
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]
    body = resp.content.decode("utf-8-sig")
    assert body.splitlines()[0].startswith("id,timestamp,actor_id")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/backend && python -m pytest tests/test_api_audit.py -v`
Expected: FAIL — 404 on `/api/admin/audit-logs` (router not registered)

- [ ] **Step 3: Implement the router**

```python
# platform/backend/app/api/audit_routes.py
"""Admin audit log query and CSV export."""
import csv
import io

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.auth.dependencies import require_admin
from app.db import get_db

router = APIRouter(prefix="/api/admin", tags=["Audit"])

_COLUMNS = ["id", "timestamp", "actor_id", "actor_username", "actor_role",
            "action", "target_type", "target_id", "detail", "ip"]


def _build_where(actor_id, action, action_prefix, date_from, date_to):
    conditions, params = [], []
    if actor_id is not None:
        conditions.append("actor_id = ?")
        params.append(actor_id)
    if action:
        conditions.append("action = ?")
        params.append(action)
    if action_prefix:
        conditions.append("action LIKE ?")
        params.append(action_prefix + "%")
    if date_from:
        conditions.append("timestamp >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("timestamp <= ?")
        params.append(date_to)
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    return where, params


@router.get("/audit-logs")
async def list_audit_logs(
    actor_id: int | None = Query(None),
    action: str | None = Query(None),
    action_prefix: str | None = Query(None),
    date_from: str | None = Query(None, alias="from"),
    date_to: str | None = Query(None, alias="to"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    admin: dict = Depends(require_admin),
):
    where, params = _build_where(actor_id, action, action_prefix, date_from, date_to)
    conn = get_db()
    total = conn.execute(f"SELECT COUNT(*) AS c FROM audit_logs{where}", params).fetchone()["c"]
    rows = conn.execute(
        f"SELECT * FROM audit_logs{where} ORDER BY id DESC LIMIT ? OFFSET ?",
        params + [page_size, (page - 1) * page_size],
    ).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows], "total": total, "page": page, "page_size": page_size}


@router.get("/audit-logs/export")
async def export_audit_logs(
    actor_id: int | None = Query(None),
    action: str | None = Query(None),
    action_prefix: str | None = Query(None),
    date_from: str | None = Query(None, alias="from"),
    date_to: str | None = Query(None, alias="to"),
    admin: dict = Depends(require_admin),
):
    where, params = _build_where(actor_id, action, action_prefix, date_from, date_to)
    conn = get_db()
    rows = conn.execute(
        f"SELECT * FROM audit_logs{where} ORDER BY id DESC LIMIT 10000", params
    ).fetchall()
    conn.close()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(_COLUMNS)
    for r in rows:
        writer.writerow([r[c] for c in _COLUMNS])
    # UTF-8 BOM so Excel opens Chinese content correctly.
    # NOTE for implementer: the string below starts with an invisible U+FEFF char —
    # in your implementation write it explicitly as "﻿" (backslash-u-f-e-f-f):
    data = "﻿" + buf.getvalue()
    return StreamingResponse(
        iter([data]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=audit-logs.csv"},
    )
```

- [ ] **Step 4: Register the router**

In `platform/backend/app/main.py`, locate the existing `app.include_router(...)` lines (e.g. for admin/auth routes) and add alongside them, following the same import style:

```python
from app.api.audit_routes import router as audit_router
app.include_router(audit_router)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd platform/backend && python -m pytest tests/test_api_audit.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add platform/backend/app/api/audit_routes.py platform/backend/app/main.py platform/backend/tests/test_api_audit.py
git commit -m "feat: add audit log query and CSV export endpoints"
```

---

## Task 3: Login/logout audit + must_change_password in auth flow

**Files:**
- Modify: `platform/backend/app/api/auth_routes.py`
- Modify: `platform/backend/app/auth/models.py` (UserOut)
- Test: `platform/backend/tests/test_api_auth_audit.py`

**Interfaces:**
- Consumes: `log_audit` (Task 1).
- Produces: `UserOut.must_change_password: bool`; `POST /api/auth/logout` → `{"status": "ok"}`; login writes `login.success` / `login.failed` audit rows with IP.

- [ ] **Step 1: Write the failing test**

```python
# platform/backend/tests/test_api_auth_audit.py
"""Login/logout audit trail + must_change_password exposure."""
from fastapi.testclient import TestClient
from app.main import app
from app.db import get_db

client = TestClient(app)


def _admin_token() -> str:
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    return resp.json()["access_token"]


def _last_audit(action):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM audit_logs WHERE action = ? ORDER BY id DESC LIMIT 1", (action,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def test_login_success_audited():
    client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    row = _last_audit("login.success")
    assert row is not None
    assert row["actor_username"] == "admin"
    assert row["ip"] != ""


def test_login_failed_audited_with_attempted_username():
    client.post("/api/auth/login", json={"username": "ghost", "password": "nope"})
    row = _last_audit("login.failed")
    assert row is not None
    assert row["actor_id"] is None
    assert "ghost" in row["detail"]


def test_login_response_contains_must_change_password():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    assert "must_change_password" in resp.json()["user"]


def test_logout_audited():
    token = _admin_token()
    resp = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    row = _last_audit("logout")
    assert row is not None and row["actor_username"] == "admin"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/backend && python -m pytest tests/test_api_auth_audit.py -v`
Expected: FAIL — no audit rows (`login.success` missing), `must_change_password` missing, 404 on logout.

- [ ] **Step 3: Implement**

In `platform/backend/app/auth/models.py`, add to `UserOut`:

```python
    must_change_password: bool = False
```

In `platform/backend/app/api/auth_routes.py`:

1. Update `_user_out` to map the new field:

```python
        must_change_password=bool(row["must_change_password"]),
```

2. Replace the `login` route (adds `Request`, audit calls, excludes deleted users):

```python
from fastapi import APIRouter, HTTPException, status, Depends, Request
from app.audit import log_audit


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, request: Request):
    ip = request.client.host if request.client else ""
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ? AND deleted_at IS NULL", (req.username,)
    ).fetchone()
    conn.close()
    if not user or not verify_password(req.password, user["password_hash"]):
        log_audit("login.failed", detail={"username": req.username}, ip=ip)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="帳號或密碼錯誤")
    if not user["is_active"]:
        log_audit("login.failed", detail={"username": req.username, "reason": "inactive"}, ip=ip)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="帳號已停用")
    token = create_token(user["id"], user["username"], user["role"])
    log_audit("login.success", actor=dict(user), ip=ip)
    return TokenResponse(access_token=token, user=_user_out(dict(user)))
```

3. Add logout:

```python
@router.post("/logout")
async def logout(request: Request, user: dict = Depends(get_current_user)):
    ip = request.client.host if request.client else ""
    log_audit("logout", actor=user, ip=ip)
    return {"status": "ok"}
```

- [ ] **Step 4: Run tests to verify pass (and no regressions)**

Run: `cd platform/backend && python -m pytest tests/test_api_auth_audit.py tests/test_api_auth.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add platform/backend/app/api/auth_routes.py platform/backend/app/auth/models.py platform/backend/tests/test_api_auth_audit.py
git commit -m "feat: audit login/logout, expose must_change_password"
```

---

## Task 4: Self-service change-password endpoint

**Files:**
- Modify: `platform/backend/app/api/auth_routes.py`
- Modify: `platform/backend/app/auth/models.py`
- Test: `platform/backend/tests/test_api_change_password.py`

**Interfaces:**
- Produces: `POST /api/auth/change-password` body `{"old_password", "new_password"}` → 200 `{"status": "ok"}`; clears `must_change_password`; audits `user.password_change`.

- [ ] **Step 1: Write the failing test**

```python
# platform/backend/tests/test_api_change_password.py
"""Self-service password change."""
from fastapi.testclient import TestClient
from app.main import app
from app.db import get_db

client = TestClient(app)


def _make_user(username, password="initpass1", must_change=1):
    admin = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"}).json()
    client.post(
        "/api/auth/register",
        json={"username": username, "password": password, "role": "student"},
        headers={"Authorization": f"Bearer {admin['access_token']}"},
    )
    conn = get_db()
    conn.execute("UPDATE users SET must_change_password = ? WHERE username = ?", (must_change, username))
    conn.commit()
    conn.close()
    resp = client.post("/api/auth/login", json={"username": username, "password": password})
    return resp.json()["access_token"]


def test_change_password_success_clears_flag_and_audits():
    token = _make_user("pwuser1")
    resp = client.post(
        "/api/auth/change-password",
        json={"old_password": "initpass1", "new_password": "newpassword9"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    # old password no longer works, new one does
    assert client.post("/api/auth/login", json={"username": "pwuser1", "password": "initpass1"}).status_code == 401
    login = client.post("/api/auth/login", json={"username": "pwuser1", "password": "newpassword9"})
    assert login.status_code == 200
    assert login.json()["user"]["must_change_password"] is False
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM audit_logs WHERE action='user.password_change' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None


def test_change_password_wrong_old():
    token = _make_user("pwuser2")
    resp = client.post(
        "/api/auth/change-password",
        json={"old_password": "WRONG", "new_password": "newpassword9"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401


def test_change_password_too_short():
    token = _make_user("pwuser3")
    resp = client.post(
        "/api/auth/change-password",
        json={"old_password": "initpass1", "new_password": "short"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/backend && python -m pytest tests/test_api_change_password.py -v`
Expected: FAIL — 404 (endpoint missing)

- [ ] **Step 3: Implement**

In `platform/backend/app/auth/models.py` add:

```python
class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str
```

In `platform/backend/app/api/auth_routes.py` add (import `ChangePasswordRequest` and `hash_password` alongside existing imports):

```python
@router.post("/change-password")
async def change_password(
    req: ChangePasswordRequest, request: Request, user: dict = Depends(get_current_user)
):
    ip = request.client.host if request.client else ""
    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="新密碼長度至少 8 碼")
    if not verify_password(req.old_password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="舊密碼錯誤")
    conn = get_db()
    conn.execute(
        "UPDATE users SET password_hash = ?, must_change_password = 0, "
        "updated_at = datetime('now') WHERE id = ?",
        (hash_password(req.new_password), user["id"]),
    )
    conn.commit()
    conn.close()
    log_audit("user.password_change", actor=user, target_type="user", target_id=user["id"], ip=ip)
    return {"status": "ok"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd platform/backend && python -m pytest tests/test_api_change_password.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add platform/backend/app/api/auth_routes.py platform/backend/app/auth/models.py platform/backend/tests/test_api_change_password.py
git commit -m "feat: add self-service change-password endpoint"
```

---

## Task 5: Soft delete + exclude deleted users everywhere

**Files:**
- Modify: `platform/backend/app/api/admin_routes.py` (delete_user, list_users, get_user)
- Modify: `platform/backend/app/auth/dependencies.py` (get_current_user)
- Test: `platform/backend/tests/test_api_soft_delete.py`

**Interfaces:**
- Produces: `DELETE /api/admin/users/{id}` → `{"deleted": true}` but row remains with `deleted_at` set; learning_events untouched.

- [ ] **Step 1: Write the failing test**

```python
# platform/backend/tests/test_api_soft_delete.py
"""Soft delete preserves data and hides the user."""
from fastapi.testclient import TestClient
from app.main import app
from app.db import get_db

client = TestClient(app)


def _admin():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _create_student(username):
    headers = _admin()
    resp = client.post(
        "/api/auth/register",
        json={"username": username, "password": "somepass1", "role": "student"},
        headers=headers,
    )
    return resp.json()["id"]


def test_soft_delete_keeps_row_and_learning_events():
    uid = _create_student("sd_user1")
    conn = get_db()
    conn.execute(
        "INSERT INTO learning_events (student_id, week, event_type, timestamp) "
        "VALUES (?, 1, 'quiz', datetime('now'))",
        (str(uid),),
    )
    conn.commit()
    resp = client.delete(f"/api/admin/users/{uid}", headers=_admin())
    assert resp.status_code == 200
    row = conn.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
    assert row is not None, "user row must be preserved"
    assert row["deleted_at"] is not None
    assert row["is_active"] == 0
    events = conn.execute(
        "SELECT COUNT(*) AS c FROM learning_events WHERE student_id = ?", (str(uid),)
    ).fetchone()["c"]
    conn.close()
    assert events == 1, "learning events must be preserved"


def test_deleted_user_hidden_and_cannot_login():
    uid = _create_student("sd_user2")
    client.delete(f"/api/admin/users/{uid}", headers=_admin())
    listing = client.get("/api/admin/users", headers=_admin()).json()
    assert all(u["id"] != uid for u in listing)
    assert client.get(f"/api/admin/users/{uid}", headers=_admin()).status_code == 404
    login = client.post("/api/auth/login", json={"username": "sd_user2", "password": "somepass1"})
    assert login.status_code == 401


def test_delete_audited():
    uid = _create_student("sd_user3")
    client.delete(f"/api/admin/users/{uid}", headers=_admin())
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM audit_logs WHERE action='user.delete' AND target_id=? "
        "ORDER BY id DESC LIMIT 1", (str(uid),)
    ).fetchone()
    conn.close()
    assert row is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/backend && python -m pytest tests/test_api_soft_delete.py -v`
Expected: FAIL — user row gone after delete (hard delete still active)

- [ ] **Step 3: Implement**

In `platform/backend/app/api/admin_routes.py`:

1. Replace `delete_user` (note: signature gains `request: Request`; import `Request` from fastapi and `log_audit` from `app.audit`):

```python
@router.delete("/users/{user_id}")
async def delete_user(user_id: int, request: Request, user=Depends(require_admin)):
    if user["id"] == user_id:
        raise HTTPException(status_code=400, detail="不能刪除自己的帳號")
    db = get_db()
    cursor = db.execute(
        "UPDATE users SET is_active = 0, deleted_at = datetime('now'), "
        "updated_at = datetime('now') WHERE id = ? AND deleted_at IS NULL",
        (user_id,),
    )
    db.commit()
    db.close()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="使用者不存在")
    ip = request.client.host if request.client else ""
    log_audit("user.delete", actor=user, target_type="user", target_id=user_id, ip=ip)
    return {"deleted": True}
```

2. In `list_users`, seed the WHERE with the soft-delete guard — replace `conditions = []` / `params = []` with:

```python
    conditions = ["deleted_at IS NULL"]
    params = []
```

and change the `where` line to always apply:

```python
    where = f" WHERE {' AND '.join(conditions)}"
```

3. In `get_user`, change the SELECT to:

```python
    row = conn.execute(
        "SELECT * FROM users WHERE id = ? AND deleted_at IS NULL", (user_id,)
    ).fetchone()
```

In `platform/backend/app/auth/dependencies.py`, change the `get_current_user` SELECT to:

```python
    user = conn.execute(
        "SELECT * FROM users WHERE id = ? AND is_active = 1 AND deleted_at IS NULL",
        (payload["sub"],),
    ).fetchone()
```

(Login already excludes deleted users from Task 3.)

- [ ] **Step 4: Run tests (new + existing admin tests)**

Run: `cd platform/backend && python -m pytest tests/test_api_soft_delete.py tests/test_api_admin.py -v`
Expected: PASS. If an existing admin test asserts hard-delete behaviour (row gone), update that assertion to expect soft delete (row present with `deleted_at`), keeping the rest of the test intact.

- [ ] **Step 5: Commit**

```bash
git add platform/backend/app/api/admin_routes.py platform/backend/app/auth/dependencies.py platform/backend/tests/test_api_soft_delete.py
git commit -m "feat: soft-delete users, preserve learning history"
```

---

## Task 6: Batch user import

**Files:**
- Modify: `platform/backend/app/api/admin_routes.py`
- Modify: `platform/backend/app/auth/models.py`
- Test: `platform/backend/tests/test_api_user_import.py`

**Interfaces:**
- Produces: `POST /api/admin/users/import` body `{"semester": str?, "rows": [{"username", "display_name"?, "email"?}]}` → `{"created": [{"username", "initial_password"}], "skipped": [{"username", "reason"}]}`. All created users get `must_change_password=1`, role `student`.

- [ ] **Step 1: Write the failing test**

```python
# platform/backend/tests/test_api_user_import.py
"""CSV batch user import."""
from fastapi.testclient import TestClient
from app.main import app
from app.db import get_db

client = TestClient(app)


def _admin():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def test_import_creates_students_with_initial_passwords():
    resp = client.post(
        "/api/admin/users/import",
        json={"semester": "115-1", "rows": [
            {"username": "s115001", "display_name": "王小明"},
            {"username": "s115002", "display_name": "李小華", "email": "b@x.tw"},
        ]},
        headers=_admin(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["created"]) == 2
    assert data["skipped"] == []
    pw = data["created"][0]["initial_password"]
    assert len(pw) >= 12
    # new account can login with the returned initial password and is flagged
    login = client.post("/api/auth/login", json={"username": "s115001", "password": pw})
    assert login.status_code == 200
    assert login.json()["user"]["must_change_password"] is True
    assert login.json()["user"]["role"] == "student"
    assert login.json()["user"]["semester"] == "115-1"


def test_import_skips_duplicates_and_bad_rows():
    client.post(
        "/api/admin/users/import",
        json={"rows": [{"username": "dupuser"}]},
        headers=_admin(),
    )
    resp = client.post(
        "/api/admin/users/import",
        json={"rows": [{"username": "dupuser"}, {"username": ""}, {"username": "okuser"}]},
        headers=_admin(),
    )
    data = resp.json()
    assert [c["username"] for c in data["created"]] == ["okuser"]
    reasons = {s["username"]: s["reason"] for s in data["skipped"]}
    assert "dupuser" in reasons and "" in reasons


def test_import_audited_with_count():
    client.post(
        "/api/admin/users/import",
        json={"rows": [{"username": "audituser1"}]},
        headers=_admin(),
    )
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM audit_logs WHERE action='user.import' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None
    assert '"created": 1' in row["detail"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/backend && python -m pytest tests/test_api_user_import.py -v`
Expected: FAIL — 404

- [ ] **Step 3: Implement**

In `platform/backend/app/auth/models.py` add:

```python
class ImportRow(BaseModel):
    username: str
    display_name: str = ""
    email: str = ""


class ImportRequest(BaseModel):
    semester: str = ""
    rows: list[ImportRow]
```

In `platform/backend/app/api/admin_routes.py` add (imports: `secrets`, `ImportRow`, `ImportRequest`, `get_setting` is already imported via `app.db` — add it if not):

```python
import secrets


@router.post("/users/import")
async def import_users(req: ImportRequest, request: Request, admin: dict = Depends(require_admin)):
    from app.db import get_setting
    semester = req.semester or get_setting("current_semester", "")
    conn = get_db()
    created, skipped = [], []
    for row in req.rows:
        username = row.username.strip()
        if not username:
            skipped.append({"username": row.username, "reason": "帳號為空"})
            continue
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            skipped.append({"username": username, "reason": "帳號已存在"})
            continue
        initial_password = secrets.token_urlsafe(9)  # 12 chars
        conn.execute(
            "INSERT INTO users (username, password_hash, display_name, email, role, "
            "semester, must_change_password) VALUES (?, ?, ?, ?, 'student', ?, 1)",
            (username, hash_password(initial_password),
             row.display_name.strip() or username, row.email.strip(), semester),
        )
        created.append({"username": username, "initial_password": initial_password})
    conn.commit()
    conn.close()
    ip = request.client.host if request.client else ""
    log_audit("user.import", actor=admin, target_type="user",
              detail={"created": len(created), "skipped": len(skipped), "semester": semester}, ip=ip)
    return {"created": created, "skipped": skipped}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd platform/backend && python -m pytest tests/test_api_user_import.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add platform/backend/app/api/admin_routes.py platform/backend/app/auth/models.py platform/backend/tests/test_api_user_import.py
git commit -m "feat: add batch user import with generated initial passwords"
```

---

## Task 7: Semester archive

**Files:**
- Modify: `platform/backend/app/api/admin_routes.py`
- Test: `platform/backend/tests/test_api_semester_archive.py`

**Interfaces:**
- Produces: `POST /api/admin/semesters/{semester}/archive` → `{"archived": int}`; deactivates all active students of that semester; audits `semester.archive`.

- [ ] **Step 1: Write the failing test**

```python
# platform/backend/tests/test_api_semester_archive.py
"""Semester batch archiving."""
from fastapi.testclient import TestClient
from app.main import app
from app.db import get_db

client = TestClient(app)


def _admin():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def test_archive_deactivates_semester_students():
    headers = _admin()
    client.post(
        "/api/admin/users/import",
        json={"semester": "old-sem", "rows": [{"username": "arch1"}, {"username": "arch2"}]},
        headers=headers,
    )
    resp = client.post("/api/admin/semesters/old-sem/archive", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["archived"] == 2
    conn = get_db()
    rows = conn.execute(
        "SELECT is_active FROM users WHERE semester='old-sem' AND role='student'"
    ).fetchall()
    conn.close()
    assert all(r["is_active"] == 0 for r in rows)


def test_archive_empty_semester_returns_zero():
    resp = client.post("/api/admin/semesters/no-such-sem/archive", headers=_admin())
    assert resp.status_code == 200
    assert resp.json()["archived"] == 0


def test_archive_audited():
    _ = _admin()
    client.post("/api/admin/semesters/no-such-sem/archive", headers=_admin())
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM audit_logs WHERE action='semester.archive' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/backend && python -m pytest tests/test_api_semester_archive.py -v`
Expected: FAIL — 404

- [ ] **Step 3: Implement**

Add to `platform/backend/app/api/admin_routes.py`:

```python
@router.post("/semesters/{semester}/archive")
async def archive_semester(semester: str, request: Request, admin: dict = Depends(require_admin)):
    conn = get_db()
    cursor = conn.execute(
        "UPDATE users SET is_active = 0, updated_at = datetime('now') "
        "WHERE semester = ? AND role = 'student' AND is_active = 1 AND deleted_at IS NULL",
        (semester,),
    )
    conn.commit()
    conn.close()
    ip = request.client.host if request.client else ""
    log_audit("semester.archive", actor=admin, target_type="semester", target_id=semester,
              detail={"archived": cursor.rowcount}, ip=ip)
    return {"archived": cursor.rowcount}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd platform/backend && python -m pytest tests/test_api_semester_archive.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add platform/backend/app/api/admin_routes.py platform/backend/tests/test_api_semester_archive.py
git commit -m "feat: add semester batch archive endpoint"
```

---

## Task 8: Audit hooks across remaining admin routes

**Files:**
- Modify: `platform/backend/app/api/admin_routes.py` (update_user, assign/remove teacher-student, settings, quiz CRUD, train-nlp, enrichment trigger)
- Modify: `platform/backend/app/api/auth_routes.py` (register)
- Test: `platform/backend/tests/test_api_audit_hooks.py`

**Interfaces:**
- Consumes: `log_audit` (Task 1). Action names from the spec catalog: `user.create`, `user.update`, `user.password_reset`, `teacher_student.assign`, `teacher_student.remove`, `settings.update`, `quiz.create`, `quiz.update`, `quiz.delete`, `nlp.train`, `enrichment.trigger`.

- [ ] **Step 1: Write the failing test**

```python
# platform/backend/tests/test_api_audit_hooks.py
"""Audit coverage of admin mutations."""
from fastapi.testclient import TestClient
from app.main import app
from app.db import get_db

client = TestClient(app)


def _admin():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _last(action):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM audit_logs WHERE action = ? ORDER BY id DESC LIMIT 1", (action,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def test_register_audited():
    client.post(
        "/api/auth/register",
        json={"username": "hookuser1", "password": "somepass1", "role": "student"},
        headers=_admin(),
    )
    row = _last("user.create")
    assert row is not None and row["actor_username"] == "admin"


def test_update_user_audits_changed_fields_not_values():
    headers = _admin()
    uid = client.post(
        "/api/auth/register",
        json={"username": "hookuser2", "password": "somepass1", "role": "student"},
        headers=headers,
    ).json()["id"]
    client.put(f"/api/admin/users/{uid}", json={"email": "new@x.tw", "password": "resetpass1"},
               headers=headers)
    row = _last("user.update")
    assert row is not None
    assert "email" in row["detail"]
    assert "resetpass1" not in row["detail"], "password value must never be logged"
    # password reset via admin also emits a dedicated event
    assert _last("user.password_reset") is not None


def test_settings_update_audited_with_keys():
    client.put("/api/admin/settings", json={"rag_top_k": "7"}, headers=_admin())
    row = _last("settings.update")
    assert row is not None and "rag_top_k" in row["detail"]


def test_quiz_crud_audited():
    headers = _admin()
    client.post(
        "/api/admin/quiz/questions",
        json={"id": "audit-q1", "week": 1, "question": "Q?", "options": ["a", "b"], "answer": 0},
        headers=headers,
    )
    assert _last("quiz.create") is not None
    client.put("/api/admin/quiz/questions/audit-q1", json={"question": "Q2?"}, headers=headers)
    assert _last("quiz.update") is not None
    client.delete("/api/admin/quiz/questions/audit-q1", headers=headers)
    assert _last("quiz.delete") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/backend && python -m pytest tests/test_api_audit_hooks.py -v`
Expected: FAIL — `_last(...)` returns None for each action

- [ ] **Step 3: Implement**

General pattern for every route below: add `request: Request` to the signature (import `Request` from fastapi if missing), compute `ip = request.client.host if request.client else ""`, call `log_audit` after the mutation succeeds.

In `platform/backend/app/api/auth_routes.py`, at the end of `register` (before `return`):

```python
    log_audit("user.create", actor=admin, target_type="user", target_id=cursor.lastrowid,
              detail={"username": req.username, "role": req.role}, ip=request.client.host if request.client else "")
```

In `platform/backend/app/api/admin_routes.py`:

- `update_user` — collect audited field names while building `updates` (field name only, never values for password). After commit:

```python
    changed = [f for f in ("display_name", "email", "role", "is_active", "semester")
               if getattr(data, f) is not None]
    ip = request.client.host if request.client else ""
    if changed:
        log_audit("user.update", actor=admin, target_type="user", target_id=user_id,
                  detail={"fields": changed}, ip=ip)
    if data.password is not None:
        log_audit("user.password_reset", actor=admin, target_type="user", target_id=user_id, ip=ip)
```

Additionally, when admin resets a password here, also set the forced-change flag — in the `data.password is not None` branch of the UPDATE builder add:

```python
        updates.append("must_change_password = 1")
```

- `assign_student_to_teacher` → `log_audit("teacher_student.assign", actor=admin, target_type="teacher_student", target_id=f"{teacher_id}:{student_id}", ip=ip)`
- `remove_student_from_teacher` → same with `"teacher_student.remove"`
- `update_settings` → `log_audit("settings.update", actor=admin, target_type="setting", detail={"keys": sorted(data.keys())}, ip=ip)`
- `admin_create_question` → `log_audit("quiz.create", actor=_user, target_type="quiz_question", target_id=body["id"], ip=ip)`
- `admin_update_question` → `"quiz.update"`, target_id=question_id
- `admin_delete_question` → `"quiz.delete"`, target_id=question_id
- `train_nlp_models` → after successful training: `log_audit("nlp.train", actor=admin, target_type="nlp", ip=ip)`
- `trigger_enrichment` → `log_audit("enrichment.trigger", actor=admin, target_type="nlp", ip=ip)`

- [ ] **Step 4: Run tests (new + full backend suite)**

Run: `cd platform/backend && python -m pytest tests/ -q`
Expected: all pass (3 pre-existing quiz-seed skips are OK)

- [ ] **Step 5: Commit**

```bash
git add platform/backend/app/api/admin_routes.py platform/backend/app/api/auth_routes.py platform/backend/tests/test_api_audit_hooks.py
git commit -m "feat: audit all admin mutations"
```

---

## Task 9: Security fixes — LLM endpoint auth, WS token required, admin seed flag

**Files:**
- Modify: `platform/backend/app/api/llm_routes.py`
- Modify: `platform/backend/app/db.py` (admin seeding)
- Modify: `render.yaml`
- Test: `platform/backend/tests/test_llm_auth.py`

**Interfaces:**
- Produces: `POST /api/llm/chat` requires Bearer token (401/403 otherwise); WS `/api/llm/ws/chat` closes with code 4401 when token missing/invalid; auto-seeded admin has `must_change_password=1`.

- [ ] **Step 1: Write the failing test**

```python
# platform/backend/tests/test_llm_auth.py
"""LLM endpoints must require authentication."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def _token():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    return resp.json()["access_token"]


def test_chat_rejects_anonymous():
    resp = client.post("/api/llm/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code in (401, 403)


def test_chat_accepts_authenticated():
    resp = client.post(
        "/api/llm/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {_token()}"},
    )
    assert resp.status_code == 200


def test_ws_rejects_missing_token():
    import pytest
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/api/llm/ws/chat"):
            pass
    assert exc_info.value.code == 4401


def test_ws_rejects_invalid_token():
    import pytest
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/api/llm/ws/chat?token=bad.token.here"):
            pass
    assert exc_info.value.code == 4401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/backend && python -m pytest tests/test_llm_auth.py -v`
Expected: FAIL — anonymous chat returns 200; WS accepts missing token

- [ ] **Step 3: Implement**

In `platform/backend/app/api/llm_routes.py`:

1. Import: `from app.auth.dependencies import get_current_user` and `from fastapi import Depends` (if missing).
2. Chat route: `async def chat(req: ChatRequest, user: dict = Depends(get_current_user)):`
3. WebSocket route — replace the token check so empty tokens are rejected too:

```python
@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket, token: str = Query(default="")):
    payload = decode_token(token) if token else None
    if not payload:
        await websocket.close(code=4401, reason="需要登入")
        return
    await websocket.accept()
```

(keep the rest of the handler unchanged)

In `platform/backend/app/db.py`, change the admin seeding INSERT to include the flag:

```python
        conn.execute(
            "INSERT INTO users (username, password_hash, display_name, role, must_change_password) "
            "VALUES (?, ?, ?, ?, 1)",
            ("admin", hash_password(settings.default_admin_password), "系統管理員", "admin"),
        )
```

Note: this INSERT runs inside `init_db()` **before** the ALTER-TABLE migrations only on brand-new DBs where executescript already created the full schema including the new columns — verify column order: the migrations run BEFORE the seeding block in init_db (semester migration precedes seeding today). Place the new ALTER migrations next to the semester one so they also precede seeding.

In `render.yaml`, add to the backend service's `envVars` list (follow the existing entry style, e.g. alongside `JWT_SECRET`):

```yaml
      - key: DEFAULT_ADMIN_PASSWORD
        sync: false
```

- [ ] **Step 4: Run tests (new + existing LLM tests)**

Run: `cd platform/backend && python -m pytest tests/test_llm_auth.py tests/test_llm.py tests/test_llm_streaming.py -v`
Expected: PASS. If existing LLM tests call `/api/llm/chat` without auth, add the admin token header to them (behaviour change is intended).

- [ ] **Step 5: Commit**

```bash
git add platform/backend/app/api/llm_routes.py platform/backend/app/db.py render.yaml platform/backend/tests/test_llm_auth.py
git commit -m "fix: require auth on LLM chat endpoints, force admin password change"
```

---

## Task 10: Frontend — ChangePasswordDialog + forced-change flow

**Files:**
- Create: `platform/frontend/src/components/auth/ChangePasswordDialog.tsx`
- Modify: `platform/frontend/src/hooks/useAuth.tsx` (User interface + expose flag)
- Modify: `platform/frontend/src/App.tsx` (render forced dialog inside the authenticated layout route)
- Test: `platform/frontend/src/components/auth/__tests__/ChangePasswordDialog.test.tsx`

**Interfaces:**
- Consumes: `POST /api/auth/change-password` (Task 4), `useAuth()` (`user`, `token`, `retryVerification`).
- Produces: `<ChangePasswordDialog forced onClose={...} />`.

- [ ] **Step 1: Write the failing test**

```tsx
// platform/frontend/src/components/auth/__tests__/ChangePasswordDialog.test.tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import ChangePasswordDialog from "../ChangePasswordDialog";

vi.mock("../../../lib/api", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../../../lib/api")>();
  return { ...mod, fetchAPI: vi.fn().mockResolvedValue({ status: "ok" }) };
});
vi.mock("../../../hooks/useAuth", () => ({
  useAuth: () => ({ token: "t", retryVerification: vi.fn() }),
}));

import { fetchAPI } from "../../../lib/api";

describe("ChangePasswordDialog", () => {
  beforeEach(() => vi.clearAllMocks());

  it("forced mode shows notice and no close button", () => {
    render(<ChangePasswordDialog forced onClose={() => {}} />);
    expect(screen.getByText(/首次登入請更換密碼/)).toBeDefined();
    expect(screen.queryByRole("button", { name: /取消/ })).toBeNull();
  });

  it("normal mode has a cancel button", () => {
    render(<ChangePasswordDialog onClose={() => {}} />);
    expect(screen.getByRole("button", { name: /取消/ })).toBeDefined();
  });

  it("rejects short new password client-side", async () => {
    render(<ChangePasswordDialog onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/舊密碼/), { target: { value: "oldpass1" } });
    fireEvent.change(screen.getByLabelText(/^新密碼/), { target: { value: "short" } });
    fireEvent.change(screen.getByLabelText(/確認新密碼/), { target: { value: "short" } });
    fireEvent.click(screen.getByRole("button", { name: /確認變更/ }));
    expect(await screen.findByText(/至少 8 碼/)).toBeDefined();
    expect(fetchAPI).not.toHaveBeenCalled();
  });

  it("submits and calls onClose on success", async () => {
    const onClose = vi.fn();
    render(<ChangePasswordDialog onClose={onClose} />);
    fireEvent.change(screen.getByLabelText(/舊密碼/), { target: { value: "oldpass1" } });
    fireEvent.change(screen.getByLabelText(/^新密碼/), { target: { value: "newpassword9" } });
    fireEvent.change(screen.getByLabelText(/確認新密碼/), { target: { value: "newpassword9" } });
    fireEvent.click(screen.getByRole("button", { name: /確認變更/ }));
    await waitFor(() => expect(onClose).toHaveBeenCalled());
    expect(fetchAPI).toHaveBeenCalledWith(
      "/api/auth/change-password",
      { old_password: "oldpass1", new_password: "newpassword9" },
      "t"
    );
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/frontend && npx vitest run src/components/auth/__tests__/ChangePasswordDialog.test.tsx`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the dialog**

```tsx
// platform/frontend/src/components/auth/ChangePasswordDialog.tsx
import { useState } from "react";
import { fetchAPI, APIError } from "../../lib/api";
import { useAuth } from "../../hooks/useAuth";

interface ChangePasswordDialogProps {
  forced?: boolean;
  onClose: () => void;
}

export default function ChangePasswordDialog({ forced = false, onClose }: ChangePasswordDialogProps) {
  const { token, retryVerification } = useAuth();
  const [oldPassword, setOldPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const submit = async () => {
    setError("");
    if (newPassword.length < 8) {
      setError("新密碼長度至少 8 碼");
      return;
    }
    if (newPassword !== confirm) {
      setError("兩次輸入的新密碼不一致");
      return;
    }
    setSubmitting(true);
    try {
      await fetchAPI("/api/auth/change-password",
        { old_password: oldPassword, new_password: newPassword }, token ?? undefined);
      await retryVerification();
      onClose();
    } catch (e) {
      if (e instanceof APIError && e.status === 401) setError("舊密碼錯誤");
      else setError("變更失敗，請稍後再試");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" role="dialog" aria-modal="true" aria-label="變更密碼">
      <div className="bg-white rounded-xl shadow-xl p-6 w-full max-w-sm space-y-4">
        <h2 className="text-lg font-bold text-gray-800">變更密碼</h2>
        {forced && (
          <p className="text-sm text-amber-600 bg-amber-50 rounded-lg p-2">
            首次登入請更換密碼後再繼續使用。
          </p>
        )}
        <div className="space-y-3">
          <label className="block text-sm text-gray-600">
            舊密碼
            <input type="password" value={oldPassword} onChange={(e) => setOldPassword(e.target.value)}
              className="mt-1 w-full border border-gray-300 rounded-lg px-3 py-2 text-sm" />
          </label>
          <label className="block text-sm text-gray-600">
            新密碼（至少 8 碼）
            <input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)}
              className="mt-1 w-full border border-gray-300 rounded-lg px-3 py-2 text-sm" />
          </label>
          <label className="block text-sm text-gray-600">
            確認新密碼
            <input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)}
              className="mt-1 w-full border border-gray-300 rounded-lg px-3 py-2 text-sm" />
          </label>
        </div>
        {error && <p className="text-sm text-red-600">{error}</p>}
        <div className="flex justify-end gap-2">
          {!forced && (
            <button onClick={onClose} className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded-lg">
              取消
            </button>
          )}
          <button onClick={submit} disabled={submitting}
            className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
            確認變更
          </button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Wire up the flag**

In `platform/frontend/src/hooks/useAuth.tsx`, add to the `User` interface:

```typescript
  must_change_password?: boolean;
```

In `platform/frontend/src/App.tsx`, inside the authenticated layout route (the `<Route>` whose `element` wraps `Home`/`WeekPage` etc. — around line 41-48): create a small gate component in the same file and render it inside that layout element:

```tsx
import { useAuth } from "./hooks/useAuth";
import ChangePasswordDialog from "./components/auth/ChangePasswordDialog";

function ForcedPasswordGate() {
  const { user } = useAuth();
  const [dismissed, setDismissed] = useState(false);
  if (!user?.must_change_password || dismissed) return null;
  return <ChangePasswordDialog forced onClose={() => setDismissed(true)} />;
}
```

Render `<ForcedPasswordGate />` adjacent to the layout's `<Outlet />` (or inside the layout element component). Import `useState` from react if not already.

- [ ] **Step 5: Run tests**

Run: `cd platform/frontend && npx vitest run src/components/auth/__tests__/ChangePasswordDialog.test.tsx`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add platform/frontend/src/components/auth/ChangePasswordDialog.tsx platform/frontend/src/components/auth/__tests__/ChangePasswordDialog.test.tsx platform/frontend/src/hooks/useAuth.tsx platform/frontend/src/App.tsx
git commit -m "feat: add change-password dialog with forced first-login mode"
```

---

## Task 11: Frontend — AuditLog admin page

**Files:**
- Create: `platform/frontend/src/pages/AuditLog.tsx`
- Modify: `platform/frontend/src/App.tsx` (route `admin/audit`)
- Modify: the admin navigation component — find it with `grep -rn "admin/users" platform/frontend/src --include="*.tsx" -l` (excluding App.tsx) and add an equivalent link to `/admin/audit` labeled 稽核紀錄, following the existing nav item markup.
- Test: `platform/frontend/src/pages/__tests__/AuditLog.test.tsx`

**Interfaces:**
- Consumes: `GET /api/admin/audit-logs` (Task 2), `fetchAPI`, `useAuth`.

- [ ] **Step 1: Write the failing test**

```tsx
// platform/frontend/src/pages/__tests__/AuditLog.test.tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import AuditLog from "../AuditLog";

const items = [
  { id: 2, timestamp: "2026-07-18 08:00:00", actor_username: "admin", actor_role: "admin",
    action: "user.create", target_type: "user", target_id: "9", detail: "{}", ip: "1.1.1.1" },
  { id: 1, timestamp: "2026-07-18 07:00:00", actor_username: "", actor_role: "",
    action: "login.failed", target_type: "", target_id: "", detail: '{"username":"ghost"}', ip: "2.2.2.2" },
];

vi.mock("../../lib/api", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../../lib/api")>();
  return {
    ...mod,
    fetchAPI: vi.fn().mockResolvedValue({ items, total: 2, page: 1, page_size: 50 }),
  };
});
vi.mock("../../hooks/useAuth", () => ({
  useAuth: () => ({ token: "t", user: { role: "admin" } }),
}));

describe("AuditLog", () => {
  it("renders audit rows from the API", async () => {
    render(<AuditLog />);
    await waitFor(() => {
      expect(screen.getByText("user.create")).toBeDefined();
      expect(screen.getByText("login.failed")).toBeDefined();
    });
  });

  it("shows tab buttons for the three views", async () => {
    render(<AuditLog />);
    expect(screen.getByRole("button", { name: /管理動作/ })).toBeDefined();
    expect(screen.getByRole("button", { name: /登入歷程/ })).toBeDefined();
    expect(screen.getByRole("button", { name: /學習行為/ })).toBeDefined();
  });

  it("shows total count", async () => {
    render(<AuditLog />);
    await waitFor(() => expect(screen.getByText(/共 2 筆/)).toBeDefined());
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/frontend && npx vitest run src/pages/__tests__/AuditLog.test.tsx`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the page**

```tsx
// platform/frontend/src/pages/AuditLog.tsx
import { useCallback, useEffect, useState } from "react";
import { fetchAPI, API_BASE } from "../lib/api";
import { useAuth } from "../hooks/useAuth";

interface AuditItem {
  id: number;
  timestamp: string;
  actor_username: string;
  actor_role: string;
  action: string;
  target_type: string;
  target_id: string;
  detail: string;
  ip: string;
}

interface AuditResponse {
  items: AuditItem[];
  total: number;
  page: number;
  page_size: number;
}

type Tab = "admin" | "login" | "learning";
const PAGE_SIZE = 50;

export default function AuditLog() {
  const { token } = useAuth();
  const [tab, setTab] = useState<Tab>("admin");
  const [page, setPage] = useState(1);
  const [actionFilter, setActionFilter] = useState("");
  const [data, setData] = useState<AuditResponse | null>(null);
  const [error, setError] = useState("");

  const buildQuery = useCallback(
    (forExport = false) => {
      const params = new URLSearchParams();
      if (tab === "login") params.set("action_prefix", "login");
      else if (tab === "admin" && actionFilter) params.set("action_prefix", actionFilter);
      if (!forExport) {
        params.set("page", String(page));
        params.set("page_size", String(PAGE_SIZE));
      }
      return params.toString();
    },
    [tab, actionFilter, page],
  );

  useEffect(() => {
    if (tab === "learning") return;
    let cancelled = false;
    setError("");
    fetchAPI<AuditResponse>(`/api/admin/audit-logs?${buildQuery()}`, undefined, token ?? undefined)
      .then((res) => { if (!cancelled) setData(res); })
      .catch(() => { if (!cancelled) setError("載入稽核紀錄失敗"); });
    return () => { cancelled = true; };
  }, [tab, page, buildQuery, token]);

  const totalPages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  const tabButton = (key: Tab, label: string) => (
    <button
      onClick={() => { setTab(key); setPage(1); }}
      className={`px-4 py-2 text-sm rounded-lg ${tab === key ? "bg-blue-600 text-white" : "bg-white text-gray-600 border border-gray-200"}`}
    >
      {label}
    </button>
  );

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-4">
      <h1 className="text-2xl font-bold text-gray-800">稽核紀錄</h1>
      <div className="flex gap-2">
        {tabButton("admin", "管理動作")}
        {tabButton("login", "登入歷程")}
        {tabButton("learning", "學習行為")}
      </div>

      {tab === "learning" ? (
        <p className="text-sm text-gray-500">
          學習行為統計請見 <a href="/dashboard" className="text-blue-600 underline">學習儀表板</a>（沿用既有分析資料）。
        </p>
      ) : (
        <>
          <div className="flex items-center gap-3">
            {tab === "admin" && (
              <select
                value={actionFilter}
                onChange={(e) => { setActionFilter(e.target.value); setPage(1); }}
                className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm"
                aria-label="動作類型篩選"
              >
                <option value="">全部動作</option>
                <option value="user">帳號管理</option>
                <option value="teacher_student">師生指派</option>
                <option value="settings">系統設定</option>
                <option value="quiz">題庫</option>
                <option value="semester">學期封存</option>
                <option value="nlp">NLP 訓練</option>
              </select>
            )}
            <span className="text-sm text-gray-500">{data ? `共 ${data.total} 筆` : ""}</span>
            <a
              href={`${API_BASE}/api/admin/audit-logs/export?${buildQuery(true)}`}
              className="ml-auto px-3 py-1.5 text-sm border border-gray-300 rounded-lg text-gray-600 hover:bg-gray-50"
              download
            >
              匯出 CSV
            </a>
          </div>

          {error && <p className="text-sm text-red-600">{error}</p>}

          <div className="overflow-x-auto border border-gray-200 rounded-lg">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50 text-left text-gray-600">
                <tr>
                  <th className="px-3 py-2">時間</th>
                  <th className="px-3 py-2">操作者</th>
                  <th className="px-3 py-2">動作</th>
                  <th className="px-3 py-2">對象</th>
                  <th className="px-3 py-2">詳情</th>
                  <th className="px-3 py-2">IP</th>
                </tr>
              </thead>
              <tbody>
                {(data?.items ?? []).map((item) => (
                  <tr key={item.id} className="border-t border-gray-100">
                    <td className="px-3 py-2 whitespace-nowrap text-gray-500">{item.timestamp}</td>
                    <td className="px-3 py-2">{item.actor_username || "—"}</td>
                    <td className="px-3 py-2 font-mono text-xs">{item.action}</td>
                    <td className="px-3 py-2 text-gray-500">
                      {item.target_type ? `${item.target_type}:${item.target_id}` : "—"}
                    </td>
                    <td className="px-3 py-2 text-gray-500 max-w-xs truncate" title={item.detail}>
                      {item.detail === "{}" ? "—" : item.detail}
                    </td>
                    <td className="px-3 py-2 text-gray-400">{item.ip || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex items-center gap-2 text-sm">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="px-3 py-1 border border-gray-300 rounded disabled:opacity-40"
            >
              上一頁
            </button>
            <span className="text-gray-500">{page} / {totalPages}</span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
              className="px-3 py-1 border border-gray-300 rounded disabled:opacity-40"
            >
              下一頁
            </button>
          </div>
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Add route + nav link**

In `platform/frontend/src/App.tsx`, next to the existing admin routes (line ~53):

```tsx
import AuditLog from "./pages/AuditLog";
// inside the layout <Route>:
<Route path="admin/audit" element={<RequireVerified><AuditLog /></RequireVerified>} />
```

Find the admin nav (`grep -rn "admin/users" platform/frontend/src --include="*.tsx" -l`, excluding App.tsx) and add a link to `/admin/audit` labeled 稽核紀錄 in the same markup style, visible to admin role only (same condition as the 帳號管理 link).

- [ ] **Step 5: Run tests**

Run: `cd platform/frontend && npx vitest run src/pages/__tests__/AuditLog.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add platform/frontend/src/pages/AuditLog.tsx platform/frontend/src/pages/__tests__/AuditLog.test.tsx platform/frontend/src/App.tsx
git add -u platform/frontend/src
git commit -m "feat: add admin audit log page with filters and CSV export"
```

---

## Task 12: Frontend — UserManagement: batch import, archive, soft-delete wording

**Files:**
- Create: `platform/frontend/src/components/admin/UserImportDialog.tsx`
- Modify: `platform/frontend/src/pages/UserManagement.tsx`
- Test: `platform/frontend/src/components/admin/__tests__/UserImportDialog.test.tsx`

**Interfaces:**
- Consumes: `POST /api/admin/users/import` (Task 6), `POST /api/admin/semesters/{sem}/archive` (Task 7).
- Produces: `<UserImportDialog onDone={() => void} onClose={() => void} />`.

- [ ] **Step 1: Write the failing test**

```tsx
// platform/frontend/src/components/admin/__tests__/UserImportDialog.test.tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import UserImportDialog from "../UserImportDialog";

vi.mock("../../../lib/api", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../../../lib/api")>();
  return {
    ...mod,
    fetchAPI: vi.fn().mockResolvedValue({
      created: [{ username: "s001", initial_password: "abcDEF123456" }],
      skipped: [{ username: "dup", reason: "帳號已存在" }],
    }),
  };
});
vi.mock("../../../hooks/useAuth", () => ({
  useAuth: () => ({ token: "t" }),
}));

import { fetchAPI } from "../../../lib/api";

describe("UserImportDialog", () => {
  beforeEach(() => vi.clearAllMocks());

  it("parses pasted CSV into a preview table", () => {
    render(<UserImportDialog onDone={() => {}} onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/名單內容/), {
      target: { value: "s001,王小明,a@x.tw\ndup,李四," },
    });
    expect(screen.getByText("王小明")).toBeDefined();
    expect(screen.getByText(/2 筆/)).toBeDefined();
  });

  it("submits rows and shows created passwords with skipped reasons", async () => {
    render(<UserImportDialog onDone={() => {}} onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/名單內容/), {
      target: { value: "s001,王小明\ndup,李四" },
    });
    fireEvent.click(screen.getByRole("button", { name: /開始匯入/ }));
    await waitFor(() => {
      expect(screen.getByText("abcDEF123456")).toBeDefined();
      expect(screen.getByText(/帳號已存在/)).toBeDefined();
    });
    expect(fetchAPI).toHaveBeenCalledWith(
      "/api/admin/users/import",
      expect.objectContaining({
        rows: [
          { username: "s001", display_name: "王小明", email: "" },
          { username: "dup", display_name: "李四", email: "" },
        ],
      }),
      "t"
    );
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/frontend && npx vitest run src/components/admin/__tests__/UserImportDialog.test.tsx`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the dialog**

```tsx
// platform/frontend/src/components/admin/UserImportDialog.tsx
import { useMemo, useState } from "react";
import { fetchAPI } from "../../lib/api";
import { useAuth } from "../../hooks/useAuth";

interface ImportResult {
  created: { username: string; initial_password: string }[];
  skipped: { username: string; reason: string }[];
}

interface UserImportDialogProps {
  onDone: () => void;
  onClose: () => void;
}

export default function UserImportDialog({ onDone, onClose }: UserImportDialogProps) {
  const { token } = useAuth();
  const [raw, setRaw] = useState("");
  const [semester, setSemester] = useState("");
  const [result, setResult] = useState<ImportResult | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const rows = useMemo(
    () =>
      raw
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => {
          const [username = "", display_name = "", email = ""] = line.split(",").map((s) => s.trim());
          return { username, display_name, email };
        }),
    [raw],
  );

  const submit = async () => {
    setError("");
    setSubmitting(true);
    try {
      const res = await fetchAPI<ImportResult>(
        "/api/admin/users/import",
        { semester, rows },
        token ?? undefined,
      );
      setResult(res);
      onDone();
    } catch {
      setError("匯入失敗，請稍後再試");
    } finally {
      setSubmitting(false);
    }
  };

  const downloadPasswords = () => {
    if (!result) return;
    // NOTE for implementer: the csv string below starts with an invisible U+FEFF
    // (Excel BOM) — in your implementation write it explicitly as "﻿" + rest.
    const csv = "﻿帳號,初始密碼\n" +
      result.created.map((c) => `${c.username},${c.initial_password}`).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "初始密碼清單.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" role="dialog" aria-modal="true" aria-label="批次匯入學生">
      <div className="bg-white rounded-xl shadow-xl p-6 w-full max-w-2xl space-y-4 max-h-[85vh] overflow-y-auto">
        <h2 className="text-lg font-bold text-gray-800">批次匯入學生</h2>

        {!result ? (
          <>
            <p className="text-sm text-gray-500">
              每行一位學生：<code className="bg-gray-100 px-1 rounded">學號,姓名,Email</code>（姓名與 Email 可省略）。
              系統會自動產生初始密碼，學生首次登入需更換。
            </p>
            <label className="block text-sm text-gray-600">
              學期（留空使用目前學期）
              <input value={semester} onChange={(e) => setSemester(e.target.value)}
                placeholder="例如 115-1"
                className="mt-1 w-full border border-gray-300 rounded-lg px-3 py-2 text-sm" />
            </label>
            <label className="block text-sm text-gray-600">
              名單內容
              <textarea value={raw} onChange={(e) => setRaw(e.target.value)} rows={8}
                placeholder={"s1150001,王小明,ming@example.com\ns1150002,李小華"}
                className="mt-1 w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono" />
            </label>
            {rows.length > 0 && (
              <div className="border border-gray-200 rounded-lg overflow-hidden">
                <div className="bg-gray-50 px-3 py-1.5 text-xs text-gray-500">預覽（{rows.length} 筆）</div>
                <table className="min-w-full text-sm">
                  <tbody>
                    {rows.slice(0, 10).map((r, i) => (
                      <tr key={i} className="border-t border-gray-100">
                        <td className="px-3 py-1.5 font-mono">{r.username || "（空）"}</td>
                        <td className="px-3 py-1.5">{r.display_name}</td>
                        <td className="px-3 py-1.5 text-gray-500">{r.email}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {rows.length > 10 && (
                  <div className="px-3 py-1.5 text-xs text-gray-400">…其餘 {rows.length - 10} 筆</div>
                )}
              </div>
            )}
            {error && <p className="text-sm text-red-600">{error}</p>}
            <div className="flex justify-end gap-2">
              <button onClick={onClose} className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded-lg">取消</button>
              <button onClick={submit} disabled={submitting || rows.length === 0}
                className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
                開始匯入
              </button>
            </div>
          </>
        ) : (
          <>
            <p className="text-sm text-gray-600">
              成功建立 {result.created.length} 筆，略過 {result.skipped.length} 筆。
              <span className="text-amber-600">初始密碼僅顯示這一次，請立即下載保存。</span>
            </p>
            {result.created.length > 0 && (
              <div className="border border-gray-200 rounded-lg overflow-hidden">
                <table className="min-w-full text-sm">
                  <thead className="bg-gray-50 text-left text-gray-600">
                    <tr><th className="px-3 py-1.5">帳號</th><th className="px-3 py-1.5">初始密碼</th></tr>
                  </thead>
                  <tbody>
                    {result.created.map((c) => (
                      <tr key={c.username} className="border-t border-gray-100">
                        <td className="px-3 py-1.5 font-mono">{c.username}</td>
                        <td className="px-3 py-1.5 font-mono">{c.initial_password}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {result.skipped.length > 0 && (
              <ul className="text-sm text-gray-500 list-disc pl-5">
                {result.skipped.map((s, i) => (
                  <li key={i}>{s.username || "（空）"}：{s.reason}</li>
                ))}
              </ul>
            )}
            <div className="flex justify-end gap-2">
              <button onClick={downloadPasswords} disabled={result.created.length === 0}
                className="px-4 py-2 text-sm border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 disabled:opacity-40">
                下載初始密碼 CSV
              </button>
              <button onClick={onClose} className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700">完成</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Wire into UserManagement**

In `platform/frontend/src/pages/UserManagement.tsx` (read the file first to match its state/handler style):

1. Import `UserImportDialog` and add `const [showImport, setShowImport] = useState(false);`
2. Next to the existing 新增使用者-style toolbar button, add:

```tsx
<button onClick={() => setShowImport(true)}
  className="px-4 py-2 text-sm bg-white border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50">
  批次匯入
</button>
```

3. Render at the end of the component: `{showImport && <UserImportDialog onDone={refreshUsers} onClose={() => setShowImport(false)} />}` where `refreshUsers` is the page's existing user-list reload function (use its actual name).
4. Change the delete button's confirm text/label to archive semantics: label 停用封存, confirm message `確定要停用封存此帳號？學習紀錄會保留，帳號將無法再登入。`
5. Add an 學期封存 toolbar button that prompts for a semester string (`window.prompt("輸入要封存的學期（例如 114-2）")`), then calls `POST /api/admin/semesters/{sem}/archive` via the page's existing authed fetch helper and shows the archived count with the page's existing feedback pattern.

- [ ] **Step 5: Run tests (dialog + existing suite)**

Run: `cd platform/frontend && npx vitest run src/components/admin/__tests__/UserImportDialog.test.tsx && npx vitest run`
Expected: PASS (all; 99 pre-existing + new)

- [ ] **Step 6: Commit**

```bash
git add platform/frontend/src/components/admin/UserImportDialog.tsx platform/frontend/src/components/admin/__tests__/UserImportDialog.test.tsx platform/frontend/src/pages/UserManagement.tsx
git commit -m "feat: batch import, semester archive, soft-delete wording in user management"
```

---

## Task 13: Final integration — full suites + build

**Files:** (no new files)

- [ ] **Step 1: Full backend suite**

Run: `cd platform/backend && python -m pytest tests/ -q`
Expected: all pass (3 pre-existing quiz-seed skips OK)

- [ ] **Step 2: Full frontend suite**

Run: `cd platform/frontend && npx vitest run`
Expected: all pass

- [ ] **Step 3: Frontend production build**

Run: `cd platform/frontend && npm run build`
Expected: build succeeds, no type errors

- [ ] **Step 4: Commit any stragglers**

```bash
git status --short
# stage only intentional leftovers, then:
git commit -m "feat: complete account management hardening and audit logging"
```
