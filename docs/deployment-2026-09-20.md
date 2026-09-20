# 2026-09-20 正式部署驗證

使用者明確授權本次 HF 重新初始化，舊資料可捨棄。未建立學生或匯入測試名單。

## 發布與儲存

- 前端已發布：[GitHub Pages](https://dong-wei-yang.github.io/ml-dl-visualization/)，對應應用修復 commit `08d04b6`。
- 後端：[HF Space](https://huggingface.co/spaces/kevin19830331/ml-dl-viz-api)，來源與實際 runtime SHA 均為 `d80e88ab996e9a98aa286195aa80bb51aee0d148`。
- [後端發布](https://github.com/DONG-WEI-YANG/ml-dl-visualization/actions/runs/35508088158)成功，Linux / Python 3.11 測試 236 passed。
- 私有 bucket `kevin19830331/ml-dl-viz-private-data` 以讀寫模式掛載 `/data`；資料庫為 `/data/app.db`。沒有購買方案或更換 cpu-basic 硬體。
- `APP_ENV=production`；HF 設定獨立隨機 `JWT_SECRET` 與強初始管理員密碼。秘密未寫入 Git 或日誌。
- 管理員帳號 `admin` 已完成首次改密碼。現行密碼使用 Windows DPAPI 加密存於本機使用者的 `.codex/secrets/ml-dl-viz-admin-current.clixml`；僅相同 Windows 使用者可解密。HF 初始密碼不等於現行登入密碼。

## 線上驗證

- `/health`：ready、database connected、RAG ready。
- 18 週各 3 題，共 54 題；第一週教材下載 HTTP 200。
- 未改密碼前管理功能 403；改密碼後原 token 401；登出後 token 401。
- [正常重啟](https://github.com/DONG-WEI-YANG/ml-dl-visualization/actions/runs/35508469382)完成後，uptime 已重設，改過的密碼仍能登入，管理員 ID 1 與改密碼狀態保留。
- [直接下載雲端 DB 查核](https://github.com/DONG-WEI-YANG/ml-dl-visualization/actions/runs/35508541674)：SQLite integrity_check 與 foreign_key_check 通過；users 1、quiz_questions 54、learning_events 0、rag_chunks 1059。下載只在臨時 runner 目錄檢查，未發布資料庫 artifact。

## 驗證邊界

本次驗證正常重啟保存，不代表突然中斷、並行多副本或長期備份演練。HF bucket 掛載採遠端同步語意，維持單副本；後續正式資料仍需 SQLite 一致性備份及還原演練。參考 [hf-mount 一致性說明](https://github.com/huggingface/hf-mount#consistency-model)。依賴掃描剩餘項目仍見原依賴修復報告，沒有宣稱所有 Python 漏洞清零。
