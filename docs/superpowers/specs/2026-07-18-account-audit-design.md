# 帳號管理強化 + 紀錄稽核 Design

**Date:** 2026-07-18
**Status:** Approved（使用者已核准，含順帶安全修復）

## Summary

強化教學平台的帳號管理（批次匯入、自助改密碼、強制改密、軟刪除、學期封存）並新增完整稽核紀錄（管理動作 + 登入歷程 + 學習行為檢視），含 Admin 稽核後台頁。順帶修復兩個健檢發現的 High 級安全問題（LLM 端點無驗證、預設 admin 弱密碼）。

## Decisions Log

| Decision | Choice | Alternatives Considered |
|----------|--------|------------------------|
| 稽核寫入方式 | 路由內明確呼叫 `log_audit()` helper | FastAPI middleware（無語意）、SQLite trigger（無操作者身分） |
| 登入歷程儲存 | 併入 `audit_logs`（action=login.*） | 獨立 login_history 表 |
| 刪除語意 | 軟刪除（`deleted_at` 標記），保留學習紀錄 | 維持硬刪除 |
| 學習行為稽核 | 沿用既有 `learning_events`，稽核頁加分頁檢視 | 複製進 audit_logs（重複儲存） |
| 初始密碼交付 | 匯入後回傳一次性密碼清單（CSV 下載） | Email 寄送（平台無 SMTP） |
| 學習行為檢視實作方式（2026-07-19 追記） | 稽核頁「學習行為」分頁改為連結既有 `/dashboard`（學習儀表板），不另建 `learning_events` 分頁檢視表格 | 依原計畫在稽核頁內建分頁檢視表格——儀表板已將 `learning_events` 視覺化，另建一份原始分頁表格增益有限，且需重複維護篩選/分頁邏輯 |

## Database

### 新表 `audit_logs`

```sql
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    actor_id INTEGER,                -- NULL for failed login of unknown user
    actor_username TEXT NOT NULL DEFAULT '',
    actor_role TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL,            -- dot-namespaced, see Action Catalog
    target_type TEXT NOT NULL DEFAULT '',  -- user|setting|quiz_question|teacher_student|semester|nlp|session
    target_id TEXT NOT NULL DEFAULT '',
    detail TEXT NOT NULL DEFAULT '{}',     -- JSON: changed fields (never passwords/hashes), counts, etc.
    ip TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_logs(actor_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);
```

### users 遷移（沿用 db.py 既有 try/except ALTER 模式）

- `deleted_at TEXT`（NULL = 未刪除）
- `must_change_password INTEGER NOT NULL DEFAULT 0`

## Action Catalog（稽核事件）

- `login.success` / `login.failed`（含 IP；failed 記嘗試的 username）/ `logout`
- `user.create` / `user.update`（detail 記變更欄位名，不記值的明碼密碼）/ `user.delete`（軟刪除）/ `user.password_reset`（admin 代改）/ `user.password_change`（自助）
- `user.import`（detail: 筆數、學期）
- `teacher_student.assign` / `teacher_student.remove`
- `semester.archive`（detail: 學期、影響人數）
- `settings.update`（detail: 變更的 key）
- `quiz.create` / `quiz.update` / `quiz.delete`
- `nlp.train` / `enrichment.trigger`

## Backend

### `app/audit/__init__.py` — helper

```python
def log_audit(action: str, *, actor: dict | None = None, target_type: str = "",
              target_id: str | int = "", detail: dict | None = None, ip: str = "") -> None
```

- 同步寫入 SQLite；失敗時 logging.warning（稽核失敗不可炸掉主要操作，但必須留 log，不可 silent pass）
- 各路由取 IP：`request.client.host`（尊重既有 proxy 設定模式）

### 端點

| Method | Path | Auth | 行為 |
|---|---|---|---|
| GET | `/api/admin/audit-logs` | admin | query: `actor_id, action, action_prefix, from, to, page, page_size(<=200)`；回 `{items, total, page}` 時間倒序 |
| GET | `/api/admin/audit-logs/export` | admin | 同篩選條件，回 CSV（UTF-8 BOM，Excel 相容） |
| POST | `/api/auth/change-password` | 登入者 | body: `{old_password, new_password}`；驗舊密碼、新密碼 ≥8 碼；成功後清 `must_change_password`；記 `user.password_change` |
| POST | `/api/auth/logout` | 登入者 | 僅記 `logout` 稽核（JWT 無伺服端狀態） |
| POST | `/api/admin/users/import` | admin | body: `{semester?, rows: [{username, display_name?, email?}]}`；為每列產 12 碼隨機初始密碼、`must_change_password=1`；重複 username 列入 `skipped`；回 `{created: [{username, initial_password}], skipped}` |
| POST | `/api/admin/semesters/{semester}/archive` | admin | 該學期 `role='student'` 全部 `is_active=0`；記 `semester.archive` |

### 既有端點修改

- `DELETE /api/admin/users/{id}` → 軟刪除：`UPDATE users SET is_active=0, deleted_at=datetime('now')`；**不再刪 learning_events / teacher_students**；記 `user.delete`
- 使用者查詢（list/get/login/get_current_user）一律排除 `deleted_at IS NOT NULL`
- `POST /api/auth/login` → 成功記 `login.success`；失敗記 `login.failed`（含 IP）；回應 `TokenResponse` 加 `must_change_password: bool`
- `POST /api/auth/register`、`PUT /users/{id}`、settings、quiz CRUD、師生指派、train-nlp、enrichment → 各補 `log_audit()`；admin 建帳號與代改密碼時設 `must_change_password=1`
- `UserOut` 加 `must_change_password`（`deleted_at` 不對外暴露——被刪除者已從所有查詢排除）

### 順帶安全修復

1. `POST /api/llm/chat` 加 `Depends(get_current_user)`；WebSocket `/api/llm/ws/chat` token 空字串或無效即拒絕（close code 4401）
2. 首次登入強制改密機制覆蓋預設 admin（`db.py` 自動建立 admin 時設 `must_change_password=1`）；`render.yaml` 加 `DEFAULT_ADMIN_PASSWORD`（sync: false 手動設定）

## Frontend

### 新頁 `pages/AuditLog.tsx`（admin only，路由 `/admin/audit`）

- 三分頁：管理動作 / 登入歷程（action_prefix=login）/ 學習行為（既有 analytics API）
- 篩選：使用者、動作類型、時間區間；分頁；CSV 匯出按鈕
- 沿用 `lib/api.ts` 的 fetchAPI（不再自建 fetch 包裝）

### `pages/UserManagement.tsx` 強化

- 「批次匯入」對話框：貼上/上傳 CSV → 預覽表格 → 送出 → 顯示結果並可下載初始密碼 CSV
- 刪除按鈕語意改「停用封存」，附說明保留學習紀錄
- 「學期封存」按鈕（選學期 → confirm → 呼叫 archive API）

### 改密碼流程

- `components/auth/ChangePasswordDialog.tsx`：登入回應或 `/me` 帶 `must_change_password=true` 時強制彈出（不可關閉）；一般情況從使用者選單開啟
- `useAuth` 儲存並暴露 `mustChangePassword` 狀態

## Error Handling

- 稽核寫入失敗：logging.warning，不阻斷主操作
- 匯入部分失敗：逐列回報 skipped 原因（重複/格式錯誤），成功列照常建立
- change-password 舊密碼錯誤：401；新密碼 <8 碼：400
- 學期封存 0 人：200 + `{archived: 0}`（非錯誤）

## Testing

- 後端（pytest，沿用 conftest 臨時 DB）：audit helper 寫入/查詢/篩選/分頁/CSV、login 成功/失敗稽核、change-password 各分支、強制改密 flag 生命週期、批次匯入（成功/重複/格式錯誤）、軟刪除保留 learning_events、學期封存、LLM 端點 401
- 前端（vitest + RTL）：AuditLog 頁渲染與篩選、匯入對話框流程、ChangePasswordDialog 強制模式

## Out of Scope

- 登入失敗鎖定（使用者未選）
- JWT 撤銷/refresh token
- Email 通知、SMTP
- learning_events 的 schema 修正（CAST JOIN 問題另案）
