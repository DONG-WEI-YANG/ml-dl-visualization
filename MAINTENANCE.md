# 維護與移交計畫 / Maintenance & Handoff Plan

## 1. 系統維護

### 1.1 定期維護項目

| 頻率 | 項目 | 指令 |
|------|------|------|
| 每日 | 檢查服務狀態 | `curl http://localhost:8000/health` |
| 每週 | 備份資料庫 | `cp data/app.db data/backup/app_$(date +%Y%m%d).db` |
| 每月 | 更新相依套件 | `pip install -U -r requirements.txt` / `npm update` |
| 每學期 | 重新訓練 NLP 模型 | 管理面板 → 訓練 NLP 模型 |
| 每學期 | 清理過期學習紀錄 | 視需求刪除舊學期資料 |

### 1.2 資料庫備份與還原

**備份：**
```bash
# 方式一：直接複製檔案（服務停止時）
cp platform/backend/data/app.db backup/app_$(date +%Y%m%d).db

# 方式二：SQL dump（服務運行中）
sqlite3 platform/backend/data/app.db ".dump" > backup/app_$(date +%Y%m%d).sql
```

**還原：**
```bash
# 從檔案還原
cp backup/app_YYYYMMDD.db platform/backend/data/app.db

# 從 SQL dump 還原
sqlite3 platform/backend/data/app.db < backup/app_YYYYMMDD.sql
```

**建議保留最近 30 天的備份。**

### 1.3 日誌監控

```bash
# 查看即時日誌
docker compose logs -f backend

# 查看錯誤
docker compose logs backend | grep ERROR
```

## 2. 常見問題排除

| 問題 | 原因 | 解決方式 |
|------|------|---------|
| 前端白屏 | JS 錯誤 | 開 DevTools Console 查看錯誤 |
| API 回傳 500 | 後端異常 | 查看 backend 日誌 |
| AI 助教無回應 | API key 失效 | 更新 `.env` 中的 API key |
| 資料庫鎖定 | 並發寫入 | 重啟 backend 服務 |
| Docker 啟動失敗 | Port 被佔用 | `docker compose down` 後重啟 |

## 3. 更新相依套件

```bash
# 後端
cd platform/backend
pip install -U -r requirements.txt
pytest tests/ -v  # 確認測試通過

# 前端
cd platform/frontend
npm update
npm test          # 確認測試通過
npx tsc --noEmit  # 確認型別正確
```

## 4. 移交清單

接手維護者需要：

- [ ] 取得程式碼存取權限
- [ ] 取得 Anthropic / OpenAI API key
- [ ] 了解 `.env` 環境變數設定
- [ ] 閱讀 `docs/architecture.md` 系統架構
- [ ] 閱讀 `platform/backend/README.md` 後端說明
- [ ] 閱讀 `platform/frontend/README.md` 前端說明
- [ ] 確認 `docker compose up -d` 可正常啟動
- [ ] 確認 `pytest` 和 `npm test` 全部通過
- [ ] 了解備份還原流程

## 5. 聯絡資訊

| 角色 | 姓名 | 備註 |
|------|------|------|
| 計畫主持人 | 楊東偉 | 系統設計與課程內容 |
