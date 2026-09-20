# 儲存與部署操作

程式與 RAG 共用 `DATABASE_PATH`；本機預設 `data/app.db`，Compose / HF image 明確使用 `/data/app.db`。Compose 使用 named volume 並唯讀掛載教材。容器以 `APP_ENV=production` 啟動，需設定獨立強秘密 `JWT_SECRET`，初次建立管理員亦需安全的 `DEFAULT_ADMIN_PASSWORD`，勿提交 `.env`。

## 首次變更正式部署前

2026-09-20 使用者明確授權本次 HF 空資料初始化，可省略舊資料備份與還原步驟。`hf-initialize.yml` 只在手動輸入 `initialize-empty` 時建立私有 bucket、設定金鑰與掛載，不刪除既有 bucket 檔案；此例外不適用未來學生資料。初始化後仍必須驗證真實掛載、應用程式寫入與重啟後保留。

1. 暫停部署及寫入，在既有正式環境找到實際 DB，記錄使用者、學習事件與題目筆數。禁止用本機 DB 覆蓋正式 DB。
2. 在現有後端目錄執行 `python scripts/backup_database.py <existing-db> <new-backup-path>`，將備份安全下載到獨立儲存。SQLite backup API 支援一致性快照；不要僅複製可能有 WAL 的主檔。
3. 演練 `python scripts/backup_database.py <backup> <new-restore-path>`，驗證業務筆數、RAG 查詢及 18 週題库，不只 integrity_check。腳本不覆蓋既有目的檔。備份包含個資與密碼雜湊，應限制存取、加密並建立保存期限。
4. 確認 HF Space 實際提供持久 `/data` 掛載，將正式備份還原至該掛載中的新路徑，驗證後設定 `DATABASE_PATH`。此項可能需要外部儲存方案或付費設定；原始碼不能完成供應商掛載。HF image 在沒有 `/data` mount 時拒絕啟動，避免靜默建立空資料庫。
5. 設定 production secrets、GitHub production environment 與 repository variable `HF_STORAGE_MIGRATION_VERIFIED=true` 才啟用部署 job。**不要先開 gate 再做備份**。目前沒有執行正式迁移、備份或部署。
6. 部署後驗證教材下載、每週非空題庫、授權與 RAG 查詢，以及重啟、重建後相同業務筆數。`/health` 的 `rag=unavailable` 表示無教材；`error` 表示索引失敗，两者皆 degraded。HTTP 200 仍可表示核心服務可用，不能單憑狀態碼認定 RAG 成功。

## 本機 Compose 驗收

先將 `.env.example` 複製為 `.env` 並設 production 強秘密，執行 `docker compose up --build -d`。確認 `/api/curriculum/week/1` 和其提供的下載連結正常。以測試環境資料驗證 `docker compose restart` 與 `docker compose up --force-recreate -d` 後資料仍在。不要使用 `docker compose down -v`，它會刪除 volume。

既有 Compose volume 原挂在 `/app/data`；相同 named volume 改挂 `/data` 不會搬移或覆蓋其內容，但仍應先備份驗證。正式 HF 舊路徑至新路徑必須獨立人工遷移，不由 CI 自動複製。
