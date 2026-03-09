# 學年度系統 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在課程平台加入學年度+學期欄位，讓學生/教師/管理員可依學期分類查詢

**Architecture:** 在 users 表加 `semester TEXT DEFAULT ''` 欄位（格式 "114-2"），system_settings 加 `current_semester`。後端 API 加篩選參數，前端帳號管理加學期選擇器和篩選器，側欄顯示學期標籤。

**Tech Stack:** FastAPI + SQLite (backend), React 19 + TypeScript + Tailwind (frontend)

---

### Task 1: DB schema + system settings

**Files:**
- Modify: `platform/backend/app/db.py`

**Changes:**
1. Add `semester TEXT NOT NULL DEFAULT ''` to users CREATE TABLE
2. Add `ALTER TABLE users ADD COLUMN semester TEXT NOT NULL DEFAULT ''` migration (safe with IF NOT EXISTS pattern)
3. Seed `current_semester` = `"114-2"` in defaults dict

---

### Task 2: Pydantic models

**Files:**
- Modify: `platform/backend/app/auth/models.py`

**Changes:**
- `UserCreate`: add `semester: str = ""`
- `UserUpdate`: add `semester: str | None = None`
- `UserOut`: add `semester: str`

---

### Task 3: Auth routes — register with semester

**Files:**
- Modify: `platform/backend/app/api/auth_routes.py`

**Changes:**
- `_user_out()`: add `semester=row["semester"]`
- `register()`: include `req.semester` in INSERT (or fallback to `current_semester` setting if empty)

---

### Task 4: Admin routes — semester filter + update + settings

**Files:**
- Modify: `platform/backend/app/api/admin_routes.py`

**Changes:**
- `_user_out()`: add `semester=row["semester"]`
- `list_users()`: add `semester: str | None = Query(None)` param, filter SQL
- `update_user()`: handle `data.semester` in updates
- `update_settings()`: add `current_semester` to allowed_keys

---

### Task 5: Analytics — semester filter

**Files:**
- Modify: `platform/backend/app/api/analytics_routes.py`
- Modify: `platform/backend/app/analytics/tracker.py`

**Changes:**
- `class_summary()` accept optional `semester` param
- When semester provided, JOIN with users table to filter by semester
- Route passes query param through

---

### Task 6: Frontend — useAuth User interface

**Files:**
- Modify: `platform/frontend/src/hooks/useAuth.tsx`

**Changes:**
- Add `semester: string` to `User` interface

---

### Task 7: Frontend — UserManagement semester support

**Files:**
- Modify: `platform/frontend/src/pages/UserManagement.tsx`

**Changes:**
- User interface: add `semester: string`
- createForm: add `semester` field, default from system settings
- Create modal: add semester selector (學年度 dropdown 110-120 + 學期 radio 上/下)
- editForm: add `semester` field
- Edit modal: add semester selector
- Filter bar: add semester dropdown filter
- Table: add 學期 column
- Stats: show per-semester counts

---

### Task 8: Frontend — Sidebar semester label

**Files:**
- Modify: `platform/frontend/src/components/Sidebar.tsx`

**Changes:**
- Format semester "114-2" → "114 學年度 下學期"
- Show below user display name

---

### Task 9: TypeScript compile check + visual verification

Run `npx tsc --noEmit` and start dev server to verify.
